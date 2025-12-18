from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, TypeVar

import numpy as np
import torch
import yaml

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(name: str = "tabtransformer_sales", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a module-level logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (parents included) if it does not exist and return the Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _absolute_path(value: str | Path, base_dir: Path) -> Path:
    """Resolve *value* relative to *base_dir* unless it is already absolute."""
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def normalize_config_paths(config: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
    """Return a copy of the config with key paths made absolute using *base_dir* as anchor."""
    paths_cfg = config.setdefault("paths", {})
    keys = [
        "raw_csv",
        "train_csv",
        "test_csv",
        "store_csv",
        "processed_dir",
        "artifacts_dir",
    ]
    for key in keys:
        value = paths_cfg.get(key)
        if value:
            paths_cfg[key] = str(_absolute_path(value, base_dir))
    return config


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load YAML from disk and ensure the top-level object is a mapping."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"YAML config not found at {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must be a mapping")
    return data


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    """Persist a JSON-serializable payload to *path* with indentation for readability."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def read_json(path: str | Path) -> Dict[str, Any]:
    """Load JSON from disk and return the parsed mapping with helpful errors."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"JSON file not found at {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON content in {path}: {exc}") from exc


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def timer(name: str, logger: Optional[logging.Logger] = None) -> Generator[None, None, None]:
    """Context manager to log elapsed time around a code block."""
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        if logger is not None:
            logger.info("%s finished in %.2fs", name, elapsed)


def resolve_processed_paths(config: Dict[str, Any]) -> tuple[Path, Path]:
    paths_cfg = config.get("paths", {})
    processed_dir = Path(paths_cfg.get("processed_dir", "data/processed"))
    processed_filename = paths_cfg.get("processed_filename", "sales_processed.parquet")
    metadata_filename = paths_cfg.get("metadata_filename", "metadata.json")
    return processed_dir / processed_filename, processed_dir / metadata_filename

def get_artifact_dir(config: Dict[str, Any], model_name: str) -> Path:
    """
    Return the artifact directory for a given model.
    If config['experiment_name'] is set, artifacts are written to:
        artifacts/<experiment_name>/<model_name>/
    Otherwise, default to:
        artifacts/<model_name>/
    """
    artifacts_root = Path(config.get("paths", {}).get("artifacts_dir", "artifacts"))
    exp_name = str(config.get("experiment_name", "")).strip()
    if exp_name:
        out_dir = artifacts_root / exp_name / model_name
    else:
        out_dir = artifacts_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def infer_device(requested: str) -> torch.device:
    """Infer the torch.device to use given a requested string such as 'cpu' or 'cuda'."""
    requested = requested.lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested in {"cuda", "gpu"}:
        logging.getLogger("tabtransformer_sales").warning(
            "CUDA requested but not available; falling back to CPU"
        )
    if requested == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


T = TypeVar("T")


class AverageMeter:
    """Track running averages for training statistics."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


def iter_prediction_rows(indices: Iterable[Any], predictions: Iterable[float]) -> Generator[Dict[str, Any], None, None]:
    """Yield dictionaries pairing index values with predictions (generator utility)."""
    for idx, pred in zip(indices, predictions):
        yield {"row_id": idx, "prediction": float(pred)}


def chunked(iterable: Iterable[T], size: int) -> Generator[List[T], None, None]:
    """Yield successive chunks from *iterable* using a while loop to satisfy coursework constraints."""
    if size <= 0:
        raise ValueError("Chunk size must be positive")
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            break
        yield batch


__all__ = [
    "AverageMeter",
    "configure_logging",
    "ensure_dir",
    "normalize_config_paths",
    "get_artifact_dir",
    "chunked",
    "infer_device",
    "iter_prediction_rows",
    "load_yaml",
    "read_json",
    "resolve_processed_paths",
    "save_json",
    "set_seed",
    "timer",
]
