from pathlib import Path

import pandas as pd
import pytest

from tabtransformer_sales.src import data


def _synthetic_frame() -> pd.DataFrame:
    records = []
    for store in [1, 2]:
        for day in range(25):
            records.append(
                {
                    "Store": store,
                    "DayOfWeek": (day % 7) + 1,
                    "Date": f"2024-01-{day + 1:02d}",
                    "Promo": day % 2,
                    "StateHoliday": "0",
                    "SchoolHoliday": 0,
                    "Customers": 100 + day + store,
                    "Sales": 1000 + day * 10 + store * 5,
                    "CompetitionDistance": 500.0,
                }
            )
    return pd.DataFrame(records)


def _config(tmp_path: Path, csv_path: Path) -> dict:
    processed_dir = tmp_path / "processed"
    return {
        "paths": {
            "raw_csv": str(csv_path),
            "processed_dir": str(processed_dir),
            "processed_filename": "sales.parquet",
            "metadata_filename": "meta.json",
            "artifacts_dir": str(tmp_path / "artifacts"),
        },
        "features": {
            "target": "Sales",
            "horizon": 1,
            "categorical": ["Store", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday"],
        },
        "splits": {"strategy": "ratios", "train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2},
    }


@pytest.mark.parametrize("save", [True, False])
def test_load_sales_data_creates_features(tmp_path: Path, save: bool) -> None:
    frame = _synthetic_frame()
    csv_path = tmp_path / "rossmann.csv"
    frame.to_csv(csv_path, index=False)
    cfg = _config(tmp_path, csv_path)
    X, y, metadata = data.load_sales_data(csv_path, cfg, save=save)
    assert not X.empty
    assert len(X) == len(y)
    assert metadata["horizon"] == 1
    assert not X.isna().any().any()
    first_target = y.iloc[0]
    assert pytest.approx(first_target) == frame.loc[1, "Sales"]
    processed_path = Path(cfg["paths"]["processed_dir"]) / cfg["paths"]["processed_filename"]
    if save:
        assert processed_path.exists()
    else:
        assert not processed_path.exists()
