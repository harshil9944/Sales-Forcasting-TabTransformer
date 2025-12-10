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
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "BaseModel":
        raise NotImplementedError


__all__ = ["BaseModel"]
