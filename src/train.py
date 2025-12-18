from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from . import data, metrics, utils
from .metrics import MetricsReport
from .models import SalesForecastTabTransformer, build_model

LOGGER = utils.configure_logging(__name__)


def _has_rows(split: tuple[pd.DataFrame, pd.Series]) -> bool:
    """Return True when the split contains at least one row."""
    X, _ = split
    return not X.empty


def train_pipeline(config: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Orchestrate dataset loading, model training, evaluation, and artifact persistence."""
    X, y, metadata = data.load_dataset_from_config(config)
    date_series = metadata.get("dates")
    splits = data.time_aware_split(X, y, date_series, config.get("splits", {}))

    model = build_model(model_name, config, metadata)
    training_cfg = config.get("training", {})

    val_data = splits.get("val") if _has_rows(splits["val"]) else None
    if isinstance(model, SalesForecastTabTransformer):
        model.fit(splits["train"][0], splits["train"][1], val_data=val_data, training_config=training_cfg)
    else:
        model.fit(splits["train"][0], splits["train"][1])

    summary: Dict[str, Dict[str, float]] = {}
    for split_name, split_data in splits.items():
        if not _has_rows(split_data):
            continue
        preds = model.predict(split_data[0])
        metric_values = metrics.evaluate_metrics(split_data[1], preds)
        summary[split_name] = metric_values
        LOGGER.info("%s split metrics:\n%s", split_name.capitalize(), MetricsReport(metric_values, label=split_name))

    artifact_dir = utils.get_artifact_dir(config, model.name)
    utils.save_json(artifact_dir / "config_snapshot.json", config)
    if isinstance(model, SalesForecastTabTransformer):
        model_path = artifact_dir / "model.pt"
    else:
        model_path = artifact_dir / "model.joblib"
    model.save(model_path)

    metrics_path = artifact_dir / "metrics.json"
    utils.save_json(metrics_path, summary)
    config_snapshot_path = artifact_dir / "config_snapshot.json"
    utils.save_json(config_snapshot_path, config)

    LOGGER.info("Training complete. Artifacts stored in %s", artifact_dir)
    report_split = summary.get("val") or summary.get("test") or summary.get("train")
    if report_split:
        LOGGER.info("\n%s", metrics.metrics_table(report_split))
    return {"artifacts": str(artifact_dir), "metrics": summary}


__all__ = ["train_pipeline"]
