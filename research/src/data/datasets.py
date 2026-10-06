"""Case trajectories and separate subject/window metadata in relative seconds."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


def numeric_seconds(values: pd.Series, name: str) -> None:
    if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        raise ValueError(f"{name} must be numeric relative seconds, never calendar datetimes")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} must be finite relative seconds")


@dataclass(frozen=True)
class TimeSeriesSchema:
    case_id_column: str
    subject_id_column: str
    time_seconds_column: str
    signal_columns: tuple[str, ...]


class CaseDataset(Protocol):
    @property
    def case_ids(self) -> tuple[str, ...]: ...
    def get_case(self, case_id: str) -> pd.DataFrame: ...


class InMemoryCaseDataset:
    """One case per trajectory; repeated operations are never concatenated."""

    def __init__(self, frame: pd.DataFrame, schema: TimeSeriesSchema) -> None:
        required = [
            schema.case_id_column,
            schema.subject_id_column,
            schema.time_seconds_column,
            *schema.signal_columns,
        ]
        if not schema.signal_columns or len(set(required)) != len(required):
            raise ValueError("Schema needs distinct case, subject, time, and signal columns")
        if frame.empty or not frame.columns.is_unique or set(required) - set(frame.columns):
            raise ValueError("Dataset is empty or has missing/duplicate columns")
        for name in (schema.case_id_column, schema.subject_id_column):
            if frame[name].isna().any() or any(
                not isinstance(v, str) or not v for v in frame[name]
            ):
                raise ValueError("Case and subject identifiers must be nonempty strings")
        numeric_seconds(frame[schema.time_seconds_column], schema.time_seconds_column)
        for _, case in frame.groupby(schema.case_id_column, sort=False):
            if case[schema.subject_id_column].nunique() != 1:
                raise ValueError("A case must map to exactly one subject")
            if not case[schema.time_seconds_column].is_monotonic_increasing:
                raise ValueError("Relative seconds must be ordered within each case")
        if any(not pd.api.types.is_numeric_dtype(frame[c]) for c in schema.signal_columns):
            raise ValueError("Signal conversion must be explicit")
        self.schema = schema
        self._frame = frame.loc[:, required].copy(deep=True)
        self._case_ids = tuple(sorted(frame[schema.case_id_column].unique()))

    @property
    def case_ids(self) -> tuple[str, ...]:
        return self._case_ids

    def get_case(self, case_id: str) -> pd.DataFrame:
        if case_id not in self._case_ids:
            raise KeyError("Unknown case identifier")
        return self._frame.loc[self._frame[self.schema.case_id_column] == case_id].copy(deep=True)


METADATA_COLUMNS = (
    "subject_id",
    "case_id",
    "anchor_time_seconds",
    "history_start_seconds",
    "history_end_seconds",
    "future_observation_end_seconds",
)


@dataclass(frozen=True)
class ClassificationDataset:
    """Aligned X/y/metadata; empty pilot outputs retain their complete schemas."""

    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame

    def __post_init__(self) -> None:
        if not self.X.columns.is_unique or not self.X.index.is_unique or self.X.shape[1] == 0:
            raise ValueError("Features require unique columns/indices and a defined schema")
        if not self.metadata.columns.is_unique:
            raise ValueError("Metadata column names must be unique")
        if not self.y.index.equals(self.X.index) or not self.metadata.index.equals(self.X.index):
            raise ValueError("X, y, and metadata must have identical row indices and order")
        if set(METADATA_COLUMNS) - set(self.metadata.columns):
            raise ValueError("Missing required subject/case/window metadata")
        if set(self.X.columns) & set(self.metadata.columns):
            raise ValueError("Metadata fields cannot appear in the model feature matrix")
        if not self.y.isin([0, 1]).all() or self.y.isna().any():
            raise ValueError("Labels must be complete binary values")
        if any(not pd.api.types.is_numeric_dtype(self.X[c]) for c in self.X.columns):
            raise ValueError("Features must be numeric")
        if np.isinf(self.X.to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError("Infinite features are unsupported")
        for name in METADATA_COLUMNS[:2]:
            if any(not isinstance(v, str) or not v for v in self.metadata[name]):
                raise ValueError("Subject/case identifiers must be nonempty strings")
        for name in METADATA_COLUMNS[2:]:
            numeric_seconds(self.metadata[name], name)
        m = self.metadata
        if (
            (m.history_start_seconds >= m.history_end_seconds)
            | (m.history_end_seconds != m.anchor_time_seconds)
            | (m.future_observation_end_seconds <= m.anchor_time_seconds)
        ).any():
            raise ValueError("Invalid history/anchor/future interval ordering")
        if not m.empty and m.groupby("case_id").subject_id.nunique().max() > 1:
            raise ValueError("A case cannot map to multiple subjects")

    @property
    def groups(self) -> pd.Series:
        return self.metadata.subject_id

    def subset(self, positions: np.ndarray) -> "ClassificationDataset":
        return ClassificationDataset(
            self.X.iloc[positions].copy(),
            self.y.iloc[positions].copy(),
            self.metadata.iloc[positions].copy(),
        )
