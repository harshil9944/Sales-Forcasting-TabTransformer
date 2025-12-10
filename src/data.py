from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from . import utils

LOGGER = utils.configure_logging(__name__)
REQUIRED_COLUMNS = {
    "Store",
    "DayOfWeek",
    "Date",
    "Promo",
    "StateHoliday",
    "SchoolHoliday",
    "Customers",
    "Sales",
    "CompetitionDistance",
}


def _maybe_merge_store_metadata(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    if "CompetitionDistance" in df.columns:
        return df
    store_path = config.get("paths", {}).get("store_csv")
    if not store_path:
        return df
    store_path = Path(store_path)
    if not store_path.exists():
        raise FileNotFoundError(f"Store metadata CSV not found at {store_path}")
    store_df = pd.read_csv(store_path)
    if "Store" not in store_df.columns:
        raise ValueError("Store metadata must contain a 'Store' column")
    if "CompetitionDistance" not in store_df.columns:
        raise ValueError("Store metadata missing required 'CompetitionDistance' column")
    merged = df.merge(store_df, on="Store", how="left")
    LOGGER.info("Merged store metadata from %s", store_path)
    return merged


def _validate_input(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Input data missing required columns: {sorted(missing)}")


def _cast_categoricals(df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def _add_time_features(df: pd.DataFrame) -> None:
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["DayOfWeek"].astype(int).isin([6, 7]).astype(int)


def _add_lag_features(df: pd.DataFrame, horizon: int) -> None:
    group = df.groupby("Store", group_keys=False)
    df["target"] = group["Sales"].shift(-horizon)
    for lag in (1, 7, 14):
        df[f"Sales_lag_{lag}"] = group["Sales"].shift(lag)
    for window in (7, 14):
        df[f"Sales_roll{window}"] = (
            group["Sales"].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )


def _sanitize_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> None:
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())
    for col in numeric_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].ffill().bfill()
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())


def _build_metadata(
    df: pd.DataFrame,
    feature_cols: List[str],
    categorical: List[str],
    numeric: List[str],
    target_column: str,
    processed_path: Path,
) -> Dict[str, Any]:
    return {
        "feature_columns": feature_cols,
        "categorical_features": categorical,
        "numeric_features": numeric,
        "target": target_column,
        "horizon": int(df.attrs.get("horizon", 1)),
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "date_min": df["Date"].min().isoformat() if not df.empty else None,
        "date_max": df["Date"].max().isoformat() if not df.empty else None,
        "processed_path": str(processed_path),
    }


def _target_output_name(features_cfg: Dict[str, Any]) -> str:
    base = features_cfg.get("target", "Sales")
    horizon = int(features_cfg.get("horizon", 1))
    suffix = "_target" if horizon == 1 else f"_target_h{horizon}"
    return f"{base}{suffix}"


def load_sales_data(csv_path: str | Path, config: Dict[str, Any], save: bool = True) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Load raw Rossmann-style CSV and engineer features for forecasting."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    df = _maybe_merge_store_metadata(df, config)
    _validate_input(df)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        raise ValueError("Date column contains invalid entries after parsing")
    df.sort_values(["Store", "Date"], inplace=True)

    features_cfg = config.get("features", {})
    target_name = features_cfg.get("target", "Sales")
    horizon = int(features_cfg.get("horizon", 1))
    target_column = _target_output_name(features_cfg)
    df.attrs["horizon"] = horizon

    _add_lag_features(df, horizon)
    _add_time_features(df)

    categorical_default = ["Store", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday"]
    categorical_cols = features_cfg.get("categorical", categorical_default)
    numeric_default = [
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
    numeric_cols = features_cfg.get("numeric", numeric_default)

    _sanitize_numeric(df, list({*numeric_cols, target_name}))

    # Drop rows without sufficient history or future target
    df = df.dropna(subset=["target"] + [col for col in numeric_cols if col in df.columns])

    df = _cast_categoricals(df, categorical_cols)

    feature_cols = categorical_cols + [col for col in numeric_cols if col in df.columns]
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing engineered features: {missing_features}")

    df[target_column] = df["target"]
    df = df.drop(columns=["target"])
    X = df[feature_cols].copy().reset_index(drop=True)
    y = df[target_column].copy().reset_index(drop=True)

    processed_path, metadata_path = utils.resolve_processed_paths(config)
    processed_df = X.copy()
    processed_df[target_column] = y
    processed_df["Date"] = df["Date"].reset_index(drop=True)

    metadata = {
        "feature_columns": feature_cols,
        "categorical_features": categorical_cols,
        "numeric_features": [col for col in numeric_cols if col in df.columns],
        "target": target_column,
        "horizon": horizon,
        "dates": processed_df["Date"],
        "stores": df["Store"].astype(str).reset_index(drop=True),
    }

    if save:
        utils.ensure_dir(processed_path.parent)
        parquet_df = processed_df.copy()
        for col in categorical_cols:
            if col in parquet_df.columns:
                parquet_df[col] = parquet_df[col].astype(str)
        parquet_df.to_parquet(processed_path, index=False)
        json_meta = _build_metadata(
            processed_df, feature_cols, categorical_cols, numeric_cols, target_column, processed_path
        )
        utils.save_json(metadata_path, json_meta)
        LOGGER.info("Saved processed dataset to %s", processed_path)

    return X, y, metadata


def load_dataset_from_config(config: Dict[str, Any], prefer_processed: bool = True) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    processed_path, metadata_path = utils.resolve_processed_paths(config)
    features_cfg = config.get("features", {})
    target_column = _target_output_name(features_cfg)
    if prefer_processed and processed_path.exists():
        df = pd.read_parquet(processed_path)
        if target_column not in df.columns:
            raise ValueError(f"Processed parquet missing target column {target_column}")
        dates = pd.to_datetime(df.get("Date")) if "Date" in df.columns else pd.Series(dtype="datetime64[ns]")
        X = df.drop(columns=[target_column] + (["Date"] if "Date" in df.columns else []))
        y = df[target_column]
        metadata = {
            "feature_columns": list(X.columns),
            "categorical_features": features_cfg.get("categorical", []),
            "numeric_features": features_cfg.get("numeric", []),
            "target": target_column,
            "horizon": features_cfg.get("horizon", 1),
            "dates": dates.reset_index(drop=True),
        }
        LOGGER.info("Loaded processed dataset from %s", processed_path)
        return X, y, metadata

    raw_path = config.get("paths", {}).get("raw_csv")
    if not raw_path:
        raise ValueError("Raw CSV path must be provided in config.paths.raw_csv")
    return load_sales_data(raw_path, config, save=True)


def time_aware_split(
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    splits_config: Dict[str, Any],
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    if dates is None or dates.empty:
        raise ValueError("Dates are required for time-based splitting")
    df = pd.DataFrame({"date": pd.to_datetime(dates).reset_index(drop=True)})
    df["idx"] = df.index
    df.sort_values("date", inplace=True)

    strategy = splits_config.get("strategy", "ratios")
    if strategy == "dates":
        train_end = pd.to_datetime(splits_config.get("train_end_date"))
        val_end = pd.to_datetime(splits_config.get("val_end_date"))
        if pd.isna(train_end) or pd.isna(val_end):
            raise ValueError("Date-based split requires train_end_date and val_end_date")
        train_idx = df[df["date"] <= train_end]["idx"].to_numpy()
        val_idx = df[(df["date"] > train_end) & (df["date"] <= val_end)]["idx"].to_numpy()
        test_idx = df[df["date"] > val_end]["idx"].to_numpy()
    else:
        ratios = (
            float(splits_config.get("train_ratio", 0.7)),
            float(splits_config.get("val_ratio", 0.15)),
            float(splits_config.get("test_ratio", 0.15)),
        )
        if not np.isclose(sum(ratios), 1.0):
            raise ValueError("Split ratios must sum to 1.0")
        n = len(df)
        train_end = int(n * ratios[0])
        val_end = train_end + int(n * ratios[1])
        train_idx = df.iloc[:train_end]["idx"].to_numpy()
        val_idx = df.iloc[train_end:val_end]["idx"].to_numpy()
        test_idx = df.iloc[val_end:]["idx"].to_numpy()

    return {
        "train": (X.iloc[train_idx].reset_index(drop=True), y.iloc[train_idx].reset_index(drop=True)),
        "val": (X.iloc[val_idx].reset_index(drop=True), y.iloc[val_idx].reset_index(drop=True)),
        "test": (X.iloc[test_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)),
    }


__all__ = [
    "load_dataset_from_config",
    "load_sales_data",
    "time_aware_split",
]
