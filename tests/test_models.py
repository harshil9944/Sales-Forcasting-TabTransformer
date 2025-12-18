
import numpy as np
import pandas as pd
import pytest

from src.models import LinearRegressionModel, SalesForecastTabTransformer, XGBoostModel

CATEGORICAL = ["Store", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday"]
NUMERIC = [
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
]


def _synthetic_features(rows: int = 20) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "Store": rng.integers(1, 5, size=rows),
        "DayOfWeek": rng.integers(1, 7, size=rows),
        "Promo": rng.integers(0, 2, size=rows),
        "StateHoliday": rng.choice(["0", "a"], size=rows),
        "SchoolHoliday": rng.integers(0, 2, size=rows),
        "Customers": rng.integers(100, 300, size=rows),
        "Sales": rng.integers(800, 1500, size=rows),
        "CompetitionDistance": rng.normal(500, 50, size=rows),
        "Sales_lag_1": rng.integers(800, 1500, size=rows),
        "Sales_lag_7": rng.integers(800, 1500, size=rows),
        "Sales_lag_14": rng.integers(800, 1500, size=rows),
        "Sales_roll7": rng.integers(800, 1500, size=rows),
        "Sales_roll14": rng.integers(800, 1500, size=rows),
        "year": 2024,
        "month": rng.integers(1, 12, size=rows),
        "day": rng.integers(1, 28, size=rows),
        "weekofyear": rng.integers(1, 52, size=rows),
        "is_weekend": rng.integers(0, 2, size=rows),
    })
    target = pd.Series(rng.integers(900, 1600, size=rows), name="target")
    return df, target


def test_linear_regression_smoke() -> None:
    X, y = _synthetic_features()
    model = LinearRegressionModel(CATEGORICAL, NUMERIC)
    model.fit(X, y)
    preds = model.predict(X.head(5))
    assert len(preds) == 5
    assert not np.isnan(preds).any()


def test_xgboost_smoke() -> None:
    pytest.importorskip("xgboost")
    X, y = _synthetic_features()
    model = XGBoostModel(CATEGORICAL, NUMERIC, params={"n_estimators": 10, "max_depth": 3})
    model.fit(X, y)
    preds = model.predict(X.head(5))
    assert len(preds) == 5


def test_tabtransformer_training_smoke() -> None:
    X, y = _synthetic_features()
    train_X, val_X = X.iloc[5:], X.iloc[:5]
    train_y, val_y = y.iloc[5:], y.iloc[:5]
    model = SalesForecastTabTransformer(
        model_config={"n_layers": 1, "n_heads": 2, "d_model": 32, "ff_dim": 64, "dropout": 0.1},
        categorical_features=CATEGORICAL,
        numeric_features=NUMERIC,
        target="target",
        device="cpu",
    )
    model.fit(
        train_X,
        train_y,
        val_data=(val_X, val_y),
        training_config={"epochs": 1, "batch_size": 8, "patience": 1, "amp": False},
    )
    preds = model.predict(val_X)
    assert len(preds) == len(val_X)
    assert not np.isnan(preds).any()

