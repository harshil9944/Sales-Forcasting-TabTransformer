from __future__ import annotations

from typing import Any, Dict

from .base import BaseModel
from .baselines import LinearRegressionModel, XGBoostModel
from .tabtransformer import SalesForecastTabTransformer


def build_model(model_name: str, config: Dict[str, Any], metadata: Dict[str, Any]) -> BaseModel:
    name = model_name.lower()
    categorical = metadata.get("categorical_features", [])
    numeric = metadata.get("numeric_features", [])
    target = metadata.get("target", "Sales")
    if name in {"linreg", "linear", "linear_regression"}:
        lin_cfg = config.get("linear_regression", {})
        return LinearRegressionModel(
            categorical_features=categorical,
            numeric_features=numeric,
            fit_intercept=bool(lin_cfg.get("fit_intercept", True)),
        )
    if name in {"xgb", "xgboost"}:
        xgb_cfg = config.get("xgboost", {})
        return XGBoostModel(
            categorical_features=categorical,
            numeric_features=numeric,
            params=xgb_cfg,
        )
    if name in {"tabtx", "tabtransformer"}:
        model_cfg = config.get("tabtransformer", {})
        device = config.get("training", {}).get("device", "cpu")
        return SalesForecastTabTransformer(
            model_config=model_cfg,
            categorical_features=categorical,
            numeric_features=numeric,
            target=target,
            device=device,
        )
    raise ValueError(f"Unknown model name: {model_name}")


__all__ = [
    "BaseModel",
    "LinearRegressionModel",
    "SalesForecastTabTransformer",
    "XGBoostModel",
    "build_model",
]
