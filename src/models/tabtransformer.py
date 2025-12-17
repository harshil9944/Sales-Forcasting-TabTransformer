from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .. import utils
from .base import BaseModel


class TabularDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Tiny dataset wrapper for categorical, numeric, and target tensors."""

    def __init__(self, cat: np.ndarray, num: np.ndarray, target: np.ndarray) -> None:
        """Convert numpy arrays into torch tensors for DataLoader consumption."""
        self.cat = torch.tensor(cat, dtype=torch.long)
        self.num = torch.tensor(num, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.float32)

    def __len__(self) -> int:
        """Number of examples in the dataset."""
        return len(self.target)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a single (categorical, numeric, target) tuple."""
        return self.cat[idx], self.num[idx], self.target[idx]


class TabTransformerEncoderLayer(nn.Module):
    """Single Transformer encoder block with attention and feed-forward sublayers."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run self-attention + feed-forward stack and return activations and attention map."""
        attn_out, attn_map = self.attn(x, x, x, need_weights=True, average_attn_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x, attn_map


class TabTransformerNet(nn.Module):
    """Full TabTransformer network that embeds categoricals and fuses numeric features."""

    def __init__(
        self,
        embedding_dims: Dict[str, Tuple[int, int]],
        d_model: int,
        n_layers: int,
        n_heads: int,
        ff_dim: int,
        dropout: float,
        numeric_dim: int,
    ) -> None:
        super().__init__()
        self.categorical_order = list(embedding_dims.keys())
        self.embeddings = nn.ModuleDict()
        self.emb_proj = nn.ModuleDict()
        for col, (num_embeddings, emb_dim) in embedding_dims.items():
            self.embeddings[col] = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=emb_dim)
            if emb_dim != d_model:
                self.emb_proj[col] = nn.Linear(emb_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.layers = nn.ModuleList(
            [TabTransformerEncoderLayer(d_model, n_heads, ff_dim, dropout) for _ in range(n_layers)]
        )
        self.attention_maps: List[torch.Tensor] = []
        self.numeric_norm = nn.LayerNorm(numeric_dim) if numeric_dim > 0 else None
        hidden = max(d_model + numeric_dim, ff_dim)
        self.head = nn.Sequential(
            nn.Linear(d_model + numeric_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, cat: torch.Tensor, num: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Perform a forward pass returning predictions and an optional attention snapshot."""
        tokens = []
        for idx, col in enumerate(self.categorical_order):
            emb = self.embeddings[col](cat[:, idx])
            if col in self.emb_proj:
                emb = self.emb_proj[col](emb)
            tokens.append(emb.unsqueeze(1))
        if tokens:
            cat_stack = torch.cat(tokens, dim=1)
            cls = self.cls_token.expand(cat_stack.shape[0], -1, -1)
            transformer_input = torch.cat([cls, cat_stack], dim=1)
        else:
            transformer_input = self.cls_token.expand(cat.shape[0], -1, -1)
        attn_snapshot: Optional[torch.Tensor] = None
        for layer in self.layers:
            transformer_input, attn_map = layer(transformer_input)
            if attn_snapshot is None:
                attn_snapshot = attn_map.detach().cpu()
        pooled = transformer_input[:, 0, :]
        if num.shape[1] > 0:
            numeric = self.numeric_norm(num) if self.numeric_norm else num
            concat = torch.cat([pooled, numeric], dim=1)
        else:
            concat = pooled
        output = self.head(concat)
        return output.squeeze(-1), attn_snapshot


class SalesForecastTabTransformer(BaseModel):
    """TabTransformer model tailored for tabular sales forecasting."""

    def __init__(
        self,
        model_config: Dict[str, Any],
        categorical_features: List[str],
        numeric_features: List[str],
        target: str = "Sales",
        device: str = "cpu",
    ) -> None:
        """Initialize a TabTransformer forecaster with configuration and feature lists."""
        super().__init__("tabtx")
        self.model_config = model_config
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.target = target
        self.device = utils.infer_device(device)
        self.category_maps: Dict[str, Dict[str, int]] = {}
        self.model: Optional[TabTransformerNet] = None
        self.numeric_scaler = StandardScaler() if numeric_features else None
        self.best_state: Optional[Dict[str, Any]] = None
        self.attention_cache: Optional[torch.Tensor] = None

    def _fit_category_maps(self, X: pd.DataFrame) -> None:
        """Build per-column vocabularies mapping category string to integer id."""
        for col in self.categorical_features:
            values = X[col].astype(str).fillna("__NA__")
            uniques = sorted(values.unique())
            self.category_maps[col] = {val: idx + 1 for idx, val in enumerate(uniques)}

    def _transform_categories(self, X: pd.DataFrame) -> np.ndarray:
        """Map categorical columns to integer ids using fitted vocabularies."""
        arrays = []
        for col in self.categorical_features:
            mapping = self.category_maps[col]
            mapped = X[col].astype(str).fillna("__NA__").map(mapping).fillna(0).astype(np.int64)
            arrays.append(mapped.to_numpy())
        if arrays:
            return np.stack(arrays, axis=1)
        return np.zeros((len(X), 1), dtype=np.int64)

    def _transform_numeric(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Scale numeric features with a shared StandardScaler; optionally fit it."""
        if not self.numeric_features:
            return np.zeros((len(X), 0), dtype=np.float32)
        data = X[self.numeric_features].astype(float).fillna(0.0)
        if self.numeric_scaler is None:
            self.numeric_scaler = StandardScaler()
        if fit:
            scaled = self.numeric_scaler.fit_transform(data)
        else:
            scaled = self.numeric_scaler.transform(data)
        return scaled.astype(np.float32)

    def _build_model(self) -> None:
        """Instantiate the TabTransformer network based on configuration and vocab sizes."""
        embedding_dims = {}
        strategy = self.model_config.get("emb_strategy", "auto")
        for col in self.categorical_features:
            vocab_size = len(self.category_maps[col]) + 1
            if strategy == "auto":
                emb_dim = min(50, max(4, int(np.sqrt(vocab_size) * 2)))
            else:
                emb_dim = int(self.model_config.get("d_model", 64))
            embedding_dims[col] = (vocab_size, emb_dim)
        self.model = TabTransformerNet(
            embedding_dims=embedding_dims,
            d_model=int(self.model_config.get("d_model", 64)),
            n_layers=int(self.model_config.get("n_layers", 2)),
            n_heads=int(self.model_config.get("n_heads", 4)),
            ff_dim=int(self.model_config.get("ff_dim", 128)),
            dropout=float(self.model_config.get("dropout", 0.1)),
            numeric_dim=len(self.numeric_features),
        ).to(self.device)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        val_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        training_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Train the TabTransformer with optional validation data and training hyperparameters."""
        if training_config is None:
            training_config = {}
        utils.set_seed(training_config.get("seed", 42))
        if not self.categorical_features:
            raise ValueError("TabTransformer requires at least one categorical feature")
        self._fit_category_maps(X)
        cat_train = self._transform_categories(X)
        num_train = self._transform_numeric(X, fit=True)
        y_train = y.to_numpy(dtype=np.float32)

        train_dataset = TabularDataset(cat_train, num_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(training_config.get("batch_size", 128)),
            shuffle=True,
        )

        val_loader = None
        if val_data is not None:
            X_val, y_val = val_data
            cat_val = self._transform_categories(X_val)
            num_val = self._transform_numeric(X_val, fit=False)
            val_dataset = TabularDataset(cat_val, num_val, y_val.to_numpy(dtype=np.float32))
            val_loader = DataLoader(val_dataset, batch_size=int(training_config.get("batch_size", 128)), shuffle=False)

        self._build_model()
        assert self.model is not None
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(training_config.get("lr", 1e-3)),
            weight_decay=float(training_config.get("weight_decay", 1e-2)),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2, factor=0.5
        )
        criterion = nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler(enabled=bool(training_config.get("amp", False)))
        best_rmse = float("inf")
        patience = int(training_config.get("patience", 5))
        epochs = int(training_config.get("epochs", 30))
        grad_clip = float(training_config.get("grad_clip", 1.0))
        no_improve = 0

        for epoch in range(epochs):
            self.model.train()
            epoch_losses = []
            for cat_batch, num_batch, target_batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    preds, _ = self.model(cat_batch.to(self.device), num_batch.to(self.device))
                    loss = criterion(preds, target_batch.to(self.device))
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                epoch_losses.append(loss.item())

            val_rmse = float(np.sqrt(np.mean(epoch_losses)))
            if val_loader is not None:
                val_rmse = self._evaluate_loader(val_loader)
                scheduler.step(val_rmse)
            else:
                scheduler.step(val_rmse)

            if val_rmse + 1e-6 < best_rmse:
                best_rmse = val_rmse
                no_improve = 0
                self.best_state = {
                    "model_state": self.model.state_dict(),
                    "category_maps": self.category_maps,
                    "numeric_scaler": self.numeric_scaler,
                }
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if self.best_state is not None and self.model is not None:
            self.model.load_state_dict(self.best_state["model_state"])

    def _evaluate_loader(self, loader: DataLoader[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> float:
        """Compute RMSE over a dataloader and optionally cache the first attention map."""
        assert self.model is not None
        self.model.eval()
        losses = []
        criterion = nn.MSELoss()
        with torch.no_grad():
            for cat_batch, num_batch, target_batch in loader:
                preds, attn = self.model(cat_batch.to(self.device), num_batch.to(self.device))
                loss = criterion(preds, target_batch.to(self.device))
                losses.append(loss.item())
                if attn is not None:
                    self.attention_cache = attn
        return float(np.sqrt(np.mean(losses))) if losses else 0.0

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions for a new feature frame using the trained network."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        cat = self._transform_categories(X)
        num = self._transform_numeric(X, fit=False)
        dataset = TabularDataset(cat, num, np.zeros(len(X), dtype=np.float32))
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        preds: List[float] = []
        self.model.eval()
        with torch.no_grad():
            for cat_batch, num_batch, _ in loader:
                pred, _ = self.model(cat_batch.to(self.device), num_batch.to(self.device))
                preds.extend(pred.cpu().numpy().tolist())
        return pd.Series(preds, index=X.index, name="prediction")

    def save(self, path: str | Path) -> None:
        """Persist model state, vocabularies, and scalers to disk."""
        if self.model is None:
            raise RuntimeError("Cannot save an untrained model")
        payload = {
            "state_dict": self.model.state_dict(),
            "model_config": self.model_config,
            "categorical_features": self.categorical_features,
            "numeric_features": self.numeric_features,
            "target": self.target,
            "category_maps": self.category_maps,
            "numeric_scaler": self.numeric_scaler,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "SalesForecastTabTransformer":
        """Load a saved TabTransformer model bundle from disk."""
        payload = torch.load(path, map_location="cpu")
        model = cls(
            payload["model_config"],
            payload["categorical_features"],
            payload["numeric_features"],
            target=payload.get("target", "Sales"),
            device=device,
        )
        model.category_maps = payload["category_maps"]
        model.numeric_scaler = payload["numeric_scaler"]
        model._build_model()
        assert model.model is not None
        model.model.load_state_dict(payload["state_dict"])
        model.model.to(model.device)
        model.model.eval()
        return model


__all__ = ["SalesForecastTabTransformer"]
