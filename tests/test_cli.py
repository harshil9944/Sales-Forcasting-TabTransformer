from pathlib import Path

import pandas as pd
import pytest
import yaml

from src import cli


def _synthetic_csv(path: Path) -> Path:
    rows = []
    base_date = pd.Timestamp("2024-01-01")
    for store in range(1, 4):
        for offset in range(18):
            current_date = base_date + pd.Timedelta(days=offset)
            rows.append(
                {
                    "Store": store,
                    "DayOfWeek": int(current_date.dayofweek + 1),
                    "Date": current_date.strftime("%Y-%m-%d"),
                    "Promo": offset % 2,
                    "StateHoliday": "0",
                    "SchoolHoliday": 0,
                    "Customers": 100 + offset + store,
                    "Sales": 1000 + 5 * store + offset * 3,
                    "CompetitionDistance": 500.0,
                }
            )
    df = pd.DataFrame(rows)
    csv_path = path / "rossmann.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _write_config(tmp_path: Path, csv_path: Path) -> Path:
    config = {
        "paths": {
            "raw_csv": str(csv_path),
            "processed_dir": str(tmp_path / "processed"),
            "processed_filename": "sales.parquet",
            "metadata_filename": "meta.json",
            "artifacts_dir": str(tmp_path / "artifacts"),
        },
        "features": {
            "target": "Sales",
            "horizon": 1,
            "categorical": ["Store", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday"],
            "numeric": [
                "Customers",
                "Sales",
                "CompetitionDistance",
                "Sales_lag_1",
                "Sales_lag_7",
                "Sales_lag_14",
                "Sales_roll7",
                "Sales_roll14",
                "year",
                "month",
                "day",
                "weekofyear",
                "is_weekend",
            ],
        },
        "splits": {"strategy": "ratios", "train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2},
        "training": {"batch_size": 8, "epochs": 2, "patience": 1, "seed": 123, "device": "cpu", "amp": False},
        "tabtransformer": {"n_layers": 1, "n_heads": 2, "d_model": 16, "ff_dim": 32, "dropout": 0.1},
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh)
    return config_path


def _run_cli(args: list[str]) -> None:
    cli.main(args)


@pytest.mark.integration
def test_cli_end_to_end(tmp_path: Path) -> None:
    csv_path = _synthetic_csv(tmp_path)
    config_path = _write_config(tmp_path, csv_path)

    _run_cli(["prepare", "--config", str(config_path), "--input", str(csv_path)])
    _run_cli(["train", "--config", str(config_path), "--model", "linreg"])
    _run_cli(["eval", "--config", str(config_path), "--model", "linreg", "--split", "test"])
    output_path = tmp_path / "preds.csv"
    _run_cli(
        [
            "predict",
            "--config",
            str(config_path),
            "--model",
            "linreg",
            "--input",
            str(csv_path),
            "--output",
            str(output_path),
        ]
    )

    processed_path = tmp_path / "processed" / "sales.parquet"
    assert processed_path.exists()
    artifact_dir = tmp_path / "artifacts" / "linreg"
    assert (artifact_dir / "model.joblib").exists()
    assert output_path.exists()
