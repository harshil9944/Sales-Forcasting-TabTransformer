import numpy as np
import pytest

from src import metrics


def test_evaluate_metrics_outputs_expected_values() -> None:
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    result = metrics.evaluate_metrics(y_true, y_pred)
    assert pytest.approx(result["MAE"], rel=1e-3) == 2.33333
    assert pytest.approx(result["RMSE"], rel=1e-3) == 2.38048
    assert pytest.approx(result["R2"], rel=1e-3) == 0.915
    assert result["MAPE"] > 0


def test_metrics_table_contains_all_metrics() -> None:
    sample = {"MAE": 1.0, "RMSE": 2.0, "R2": 0.9, "MAPE": 10.0}
    table = metrics.metrics_table(sample)
    for key in ["MAE", "RMSE", "R2", "MAPE"]:
        assert key in table
