"""A service boundary shared by a future API and demo UI."""

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class PredictionResult:
    """Probability columns carry explicit class labels; index aligns with inputs."""

    probabilities: pd.DataFrame
    model_version: str
    feature_schema_version: str


class InferenceService(Protocol):
    """Serve a fitted pipeline using training-time schemas and preprocessing.

    This feature-table boundary is intentionally transport independent. Raw
    streaming input, persistence, authentication, and UI are future work.
    """

    def predict(self, features: pd.DataFrame) -> PredictionResult: ...
