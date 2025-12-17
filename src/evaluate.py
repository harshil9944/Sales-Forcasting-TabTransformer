from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import data, metrics, utils
from .metrics import MetricsReport
from .models import LinearRegressionModel, SalesForecastTabTransformer, XGBoostModel

LOGGER = utils.configure_logging(__name__)


def load_trained_model(model_name: str, artifact_dir: Path, config: Dict[str, Any]):
    """Load a persisted model artifact for the given name from the artifact directory."""
    name = model_name.lower()
    if name in {"linreg", "linear", "linear_regression"}:
        path = artifact_dir / "model.joblib"
        return LinearRegressionModel.load(path)
    if name in {"xgb", "xgboost"}:
        path = artifact_dir / "model.joblib"
        return XGBoostModel.load(path)
    path = artifact_dir / "model.pt"
    device = config.get("training", {}).get("device", "cpu")
    return SalesForecastTabTransformer.load(path, device=device)


def _plot_scatter(y_true: pd.Series, y_pred: pd.Series, output: Path) -> None:
    """Plot predicted vs. actual scatter and save to *output* path."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, s=12, c="#1f77b4")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "--", color="gray")
    ax.set_xlabel("Actual Sales")
    ax.set_ylabel("Predicted Sales")
    ax.set_title("Predicted vs Actual")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _plot_residuals(y_true: pd.Series, y_pred: pd.Series, output: Path) -> None:
    """Plot residual histogram to help inspect model error distribution."""
    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=30, color="#ff7f0e", alpha=0.7)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Distribution")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _plot_attention_heatmap(model: SalesForecastTabTransformer, output: Path, categorical: list[str]) -> None:
    """Render the first-layer attention heatmap when available from the TabTransformer."""
    if model.attention_cache is None:
        return
    attn = model.attention_cache.mean(dim=1).squeeze(0).numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(attn, cmap="viridis")
    tokens = ["CLS", *categorical]
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45)
    ax.set_yticklabels(tokens)
    ax.set_title("Attention Heatmap (Layer 1)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def evaluate_pipeline(config: Dict[str, Any], model_name: str, split: str = "test") -> Dict[str, Any]:
    """Evaluate a trained model on a dataset split, plot diagnostics, and save metrics."""
    X, y, metadata = data.load_dataset_from_config(config)
    splits = data.time_aware_split(X, y, metadata.get("dates"), config.get("splits", {}))
    if split not in splits:
        raise ValueError(f"Unknown split {split}")
    X_split, y_split = splits[split]
    artifact_dir = utils.get_artifact_dir(config, model_name.lower())
    model = load_trained_model(model_name, artifact_dir, config)
    preds = model.predict(X_split)
    metric_values = metrics.evaluate_metrics(y_split, preds)

    utils.save_json(artifact_dir / f"metrics_{split}.json", {split: metric_values})
    _plot_scatter(y_split, preds, artifact_dir / "pred_vs_actual.png")
    _plot_residuals(y_split, preds, artifact_dir / "residual_hist.png")

    training_cfg = config.get("training", {})
    if training_cfg.get("plot_attention", False) and isinstance(model, SalesForecastTabTransformer):
        _plot_attention_heatmap(model, artifact_dir / "attention_heatmap.png", metadata.get("categorical_features", []))

    LOGGER.info("Evaluation metrics (%s):\n%s", split, MetricsReport(metric_values, label=split))
    return metric_values


__all__ = ["evaluate_pipeline", "load_trained_model"]
