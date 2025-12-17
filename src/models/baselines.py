from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Iterable, List

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .base import BaseModel

try:
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover - handled in tests if package missing
    XGBRegressor = None  # type: ignore
    _xgb_import_error = exc
else:
    _xgb_import_error = None


def _one_hot_encoder() -> OneHotEncoder:
    """Create a OneHotEncoder that works across sklearn versions (handles sparse_output rename)."""
    params: Dict[str, Any] = {"handle_unknown": "ignore"}
    signature = inspect.signature(OneHotEncoder.__init__)
    if "sparse_output" in signature.parameters:
        params["sparse_output"] = False
    else:
        params["sparse"] = False
    return OneHotEncoder(**params)


class _PreprocessorFactory:
    """Helper to create reusable sklearn preprocessing pipelines."""
    @staticmethod
    def make(categorical: Iterable[str], numeric: Iterable[str]) -> ColumnTransformer:
        """Build a ColumnTransformer with categorical encoders and numeric scalers."""
        cat_features = list(categorical)
        num_features = list(numeric)
        transformers = []
        if cat_features:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", _one_hot_encoder()),
                        ]
                    ),
                    cat_features,
                )
            )
        if num_features:
            transformers.append(
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    num_features,
                )
            )
        return ColumnTransformer(transformers=transformers, remainder="drop")


class LinearRegressionModel(BaseModel):
    """Baseline linear regression with one-hot encoding."""

    def __init__(
        self,
        categorical_features: List[str],
        numeric_features: List[str],
        fit_intercept: bool = True,
    ) -> None:
        super().__init__("linreg")
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        preprocessor = _PreprocessorFactory.make(categorical_features, numeric_features)
        self.pipeline = Pipeline(
            steps=[
                ("pre", preprocessor),
                ("regressor", LinearRegression(fit_intercept=fit_intercept)),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, **_: Any) -> None:
        """Train the linear regression pipeline on the provided frame."""
        self.pipeline.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Return predictions as a pandas Series aligned to input indices."""
        preds = self.pipeline.predict(X)
        return pd.Series(preds, index=X.index, name="prediction")

    def save(self, path: str | Path) -> None:
        """Persist pipeline and feature metadata to disk via joblib."""
        payload = {
            "categorical_features": self.categorical_features,
            "numeric_features": self.numeric_features,
            "pipeline": self.pipeline,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path, **_: Any) -> "LinearRegressionModel":
        """Reconstruct a saved LinearRegressionModel from disk."""
        payload = joblib.load(path)
        model = cls(payload["categorical_features"], payload["numeric_features"])
        model.pipeline = payload["pipeline"]
        return model


class XGBoostModel(BaseModel):
    """XGBoost regressor with preprocessing pipeline."""

    def __init__(
        self,
        categorical_features: List[str],
        numeric_features: List[str],
        params: Dict[str, Any],
    ) -> None:
        if XGBRegressor is None:
            raise ImportError("xgboost is required for XGBoostModel") from _xgb_import_error
        super().__init__("xgb")
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        preprocessor = _PreprocessorFactory.make(categorical_features, numeric_features)
        model_params = {
            "objective": "reg:squarederror",
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "verbosity": 0,
        }
        model_params.update(params)
        self.pipeline = Pipeline(
            steps=[
                ("pre", preprocessor),
                (
                    "regressor",
                    XGBRegressor(**model_params),  # type: ignore[arg-type]
                ),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, **_: Any) -> None:
        """Train the XGBoost pipeline on the provided frame."""
        self.pipeline.fit(X, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Return predictions as a pandas Series aligned to input indices."""
        preds = self.pipeline.predict(X)
        return pd.Series(preds, index=X.index, name="prediction")

    def save(self, path: str | Path) -> None:
        """Persist pipeline and feature metadata to disk via joblib."""
        payload = {
            "categorical_features": self.categorical_features,
            "numeric_features": self.numeric_features,
            "pipeline": self.pipeline,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path, **_: Any) -> "XGBoostModel":
        """Reconstruct a saved XGBoostModel from disk."""
        payload = joblib.load(path)
        model = cls(payload["categorical_features"], payload["numeric_features"], params={})
        model.pipeline = payload["pipeline"]
        return model


__all__ = ["LinearRegressionModel", "XGBoostModel"]
