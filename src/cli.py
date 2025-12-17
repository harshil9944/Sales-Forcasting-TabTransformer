from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from . import data, evaluate, train, utils
from .metrics import metrics_table

LOGGER = utils.configure_logging(__name__)


def _load_config(path: str) -> dict[str, Any]:
    """Load and normalize YAML config paths relative to the project root or config file location."""
    config_path = Path(path).resolve()
    config = utils.load_yaml(config_path)
    project_root = Path(__file__).resolve().parents[1]
    base_dir = project_root if project_root in config_path.parents else config_path.parent
    return utils.normalize_config_paths(config, base_dir)


def cmd_prepare(args: argparse.Namespace) -> None:
    """CLI entrypoint for dataset preparation from raw CSV."""
    config = _load_config(args.config)
    raw_path = args.input or config.get("paths", {}).get("raw_csv")
    if not raw_path:
        raise ValueError("Raw CSV path required via --input or config.paths.raw_csv")
    X, y, metadata = data.load_sales_data(raw_path, config, save=True)
    LOGGER.info(
        "Prepared dataset with %d rows, %d features, horizon=%s",
        len(X),
        X.shape[1],
        metadata.get("horizon"),
    )


def cmd_train(args: argparse.Namespace) -> None:
    """CLI entrypoint for training a configured model."""
    config = _load_config(args.config)
    result = train.train_pipeline(config, args.model)
    LOGGER.info("Artifacts written to %s", result["artifacts"])


def cmd_eval(args: argparse.Namespace) -> None:
    """CLI entrypoint for evaluating a trained model on a requested split."""
    config = _load_config(args.config)
    metric_values = evaluate.evaluate_pipeline(config, args.model, split=args.split)
    LOGGER.info("%s", metrics_table(metric_values))


def cmd_predict(args: argparse.Namespace) -> None:
    """CLI entrypoint for running batch inference and writing predictions to CSV."""
    config = _load_config(args.config)
    artifact_dir = utils.get_artifact_dir(config, args.model.lower())
    model = evaluate.load_trained_model(args.model, artifact_dir, config)
    if args.input:
        X, _, _ = data.load_sales_data(args.input, config, save=False)
    else:
        X, _, _ = data.load_dataset_from_config(config)
    preds = model.predict(X)
    output_path = Path(args.output or artifact_dir / "predictions.csv")
    rows = list(utils.iter_prediction_rows(preds.index if hasattr(preds, "index") else range(len(preds)), preds))
    df_out = pd.DataFrame(rows)
    preview = next(utils.chunked(rows, 5), [])
    if preview:
        LOGGER.info("Preview predictions: %s", preview)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    LOGGER.info("Saved predictions to %s", output_path)


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argparse parser and subcommands."""
    parser = argparse.ArgumentParser(description="TabTransformer sales forecasting CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare", help="Process raw CSV and save parquet")
    prep.add_argument("--config", default="config/default.yaml", help="Path to YAML config")
    prep.add_argument("--input", help="Optional raw CSV path overriding config")
    prep.set_defaults(func=cmd_prepare)

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config")
    train_parser.add_argument("--model", choices=["linreg", "xgb", "tabtx"], default="tabtx")
    train_parser.set_defaults(func=cmd_train)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained model")
    eval_parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config")
    eval_parser.add_argument("--model", choices=["linreg", "xgb", "tabtx"], default="tabtx")
    eval_parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    eval_parser.set_defaults(func=cmd_eval)

    predict_parser = subparsers.add_parser("predict", help="Run inference using a trained model")
    predict_parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config")
    predict_parser.add_argument("--model", choices=["linreg", "xgb", "tabtx"], default="tabtx")
    predict_parser.add_argument("--input", help="CSV file with raw features")
    predict_parser.add_argument("--output", help="Where to write predictions CSV")
    predict_parser.set_defaults(func=cmd_predict)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
