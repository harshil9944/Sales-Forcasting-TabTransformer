from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


METRIC_ORDER = ("MAE", "RMSE", "R2", "MAPE")


def evaluate_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    """Compute standard regression metrics (MAE, RMSE, R2, MAPE) from true and predicted arrays."""
    y_true_arr = np.asarray(list(y_true), dtype=np.float64)
    y_pred_arr = np.asarray(list(y_pred), dtype=np.float64)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    r2 = float(r2_score(y_true_arr, y_pred_arr))
    denom = np.where(y_true_arr == 0, np.nan, np.abs(y_true_arr))
    mape = float(np.nanmean(np.abs((y_true_arr - y_pred_arr) / denom)) * 100)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def metrics_table(metrics: Dict[str, float]) -> str:
    """Render a small Markdown table from a metrics dictionary."""
    header = "| Metric | Value |\n| --- | --- |"
    rows = [f"| {name} | {metrics[name]:.4f} |" for name in METRIC_ORDER if name in metrics]
    return "\n".join([header, *rows])


@dataclass
class MetricsReport:
    """Container supporting operator overloading and human-readable rendering of metrics."""

    values: Dict[str, float]
    label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.values, dict):
            raise TypeError("values must be a dictionary of metric name to float")

    def __add__(self, other: "MetricsReport") -> "MetricsReport":
        merged: Dict[str, float] = {}
        keys = set(self.values) | set(other.values)
        for key in keys:
            merged[key] = self.values.get(key, 0.0) + other.values.get(key, 0.0)
        label = self.label if self.label is not None else other.label
        return MetricsReport(merged, label=label)

    def __truediv__(self, scalar: float) -> "MetricsReport":
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide metrics by zero")
        scaled = {k: v / scalar for k, v in self.values.items()}
        return MetricsReport(scaled, label=self.label)

    def __str__(self) -> str:  # pragma: no cover - formatting helper
        heading = f"### {self.label} Metrics\n" if self.label else ""
        return heading + metrics_table(self.values)

    def to_dict(self) -> Dict[str, float]:
        """Return the underlying metric dictionary."""
        return dict(self.values)


__all__ = ["MetricsReport", "evaluate_metrics", "metrics_table"]
