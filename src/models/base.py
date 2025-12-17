from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd


class BaseModel(ABC):
    """Abstract interface for sales forecasting models."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> None:
        """Train the model on features X and targets y."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions for the provided feature matrix."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the trained model to disk."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "BaseModel":
        """Load a saved model instance from disk."""
        raise NotImplementedError


__all__ = ["BaseModel"]
