"""History-only, relative-second feature extraction interface."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from intraop.data.datasets import numeric_seconds


@dataclass(frozen=True)
class Window:
    case_id: str
    subject_id: str
    anchor_time_seconds: float
    history_start_seconds: float
    history_end_seconds: float
    history: pd.DataFrame
    raw_samples: dict[str, pd.DataFrame] = field(default_factory=dict)
    future_observation_end_seconds: float | None = None

    def __post_init__(self) -> None:
        times = (self.history_start_seconds, self.history_end_seconds, self.anchor_time_seconds)
        if any(type(t) not in (int, float) or not np.isfinite(t) for t in times):
            raise ValueError("Window times must be numeric relative seconds")
        if self.history_start_seconds >= self.history_end_seconds:
            raise ValueError("Window history must have positive length")
        if self.history_end_seconds != self.anchor_time_seconds:
            raise ValueError("History must end at the anchor")
        if self.future_observation_end_seconds is not None:
            end = self.future_observation_end_seconds
            if (
                type(end) not in (int, float)
                or not np.isfinite(end)
                or end <= self.anchor_time_seconds
            ):
                raise ValueError("Future observation bound must be numeric and follow the anchor")
        for frame in (self.history, *self.raw_samples.values()):
            numeric_seconds(frame.time_seconds, "time_seconds")
            if (
                (frame.time_seconds <= self.history_start_seconds)
                | (frame.time_seconds > self.anchor_time_seconds)
            ).any():
                raise ValueError("Feature inputs must lie entirely in (history_start, anchor]")


class FeatureExtractor(Protocol):
    @property
    def feature_names(self) -> tuple[str, ...]: ...
    def extract(self, window: Window) -> Mapping[str, float]: ...


class PlaceholderFeatureExtractor:
    @property
    def feature_names(self) -> tuple[str, ...]:
        raise NotImplementedError("Use the explicit VitalDB core extractor for v0.1")

    def extract(self, window: Window) -> Mapping[str, float]:
        raise NotImplementedError("Feature extraction has not been implemented")
