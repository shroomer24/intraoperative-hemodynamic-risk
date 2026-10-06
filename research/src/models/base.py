"""Shared classification API for future TabPFN, XGBoost, and linear adapters."""

from typing import Protocol

import numpy as np
import pandas as pd


class ClassificationModel(Protocol):
    """Fit on training rows only; inference must preserve feature schema/order.

    classes_ determines probability column order. Adapters must document fitted
    state, missingness handling, serialization, and deterministic seed support.
    Any learned encoder, imputer, scaler, or selector belongs inside a pipeline
    fitted exclusively on the training partition.
    """

    @property
    def classes_(self) -> np.ndarray: ...

    def fit(self, X: pd.DataFrame, y: pd.Series, *, seed: int) -> "ClassificationModel": ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


class PlaceholderClassifier:
    """Fail explicitly until a backend is chosen and implemented."""

    @property
    def classes_(self) -> np.ndarray:
        raise NotImplementedError("No classifier backend is implemented")

    def fit(self, X: pd.DataFrame, y: pd.Series, *, seed: int) -> "PlaceholderClassifier":
        raise NotImplementedError("Model backend implementation is intentionally deferred")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("No classifier backend is implemented")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError("No classifier backend is implemented")
