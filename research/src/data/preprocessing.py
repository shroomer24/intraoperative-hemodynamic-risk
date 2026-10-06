"""Bounded causal sample-and-hold on each case's surgical integer-second grid."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from intraop.data.datasets import numeric_seconds
from intraop.data.protocol import MAP_TECHNICAL_CEILING, MAX_AGE_SECONDS, TRACKS
from intraop.data.vitaldb_reader import CaseSignals


@dataclass(frozen=True)
class ProcessedCase:
    case_id: str
    subject_id: str
    opstart_seconds: float
    opend_seconds: float
    grid: pd.DataFrame
    raw_valid_samples: dict[str, pd.DataFrame]
    audit: dict


def valid_samples(samples: pd.DataFrame, *, signal: str) -> pd.DataFrame:
    numeric_seconds(samples.time_seconds, "time_seconds")
    mask = np.isfinite(samples.value)
    if signal == "map":
        mask &= (samples.value > 0) & (samples.value <= MAP_TECHNICAL_CEILING)
    return samples.loc[mask].sort_values("time_seconds", kind="stable").copy()


def causal_hold(samples: pd.DataFrame, grid_seconds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Latest valid sample <= u; age <=10 inclusive. Duplicates: last in file order.

    samples must already be validity-filtered. Age remains recorded even when
    stale; values become missing. A later invalid sample never resets freshness.
    """
    times = samples.time_seconds.to_numpy(dtype=float)
    values = samples.value.to_numpy(dtype=float)
    if not np.isfinite(times).all() or not np.isfinite(values).all() or (np.diff(times) < 0).any():
        raise ValueError("Causal hold expects ordered, finite valid samples")
    grid_seconds = np.asarray(grid_seconds, dtype=float)
    if not np.isfinite(grid_seconds).all() or (np.diff(grid_seconds) <= 0).any():
        raise ValueError("Grid seconds must be finite and strictly increasing")
    result = np.full(len(grid_seconds), np.nan)
    age = np.full(len(grid_seconds), np.nan)
    if not len(times):
        return result, age
    indices = np.searchsorted(times, grid_seconds, side="right") - 1
    available = indices >= 0
    age[available] = grid_seconds[available] - times[indices[available]]
    fresh = available & (age <= MAX_AGE_SECONDS)
    result[fresh] = values[indices[fresh]]
    return result, age


def preprocess_case(case: CaseSignals) -> ProcessedCase:
    if not np.isfinite([case.opstart_seconds, case.opend_seconds]).all():
        raise ValueError("Surgical bounds must be finite")
    if not case.opstart_seconds < case.opend_seconds:
        raise ValueError("Invalid surgical interval")
    end = min(
        case.opend_seconds,
        case.recording_end_seconds
        if case.recording_end_seconds is not None
        else case.opend_seconds,
    )
    seconds = np.arange(max(0, np.ceil(case.opstart_seconds)), np.floor(end) + 1, dtype=float)
    frame = pd.DataFrame({"time_seconds": seconds})
    raw = {}
    invalid = {}
    for signal in TRACKS:
        original = case.samples.get(
            signal, pd.DataFrame(columns=["time_seconds", "value"], dtype=float)
        )
        # A fresh earlier observation can legitimately hold into the surgical
        # grid. Only the represented seconds, not the source samples, are clipped
        # at opstart. Never extend represented observation beyond recording end.
        original = original.loc[original.time_seconds <= end]
        clean = valid_samples(original, signal=signal)
        raw[signal] = clean
        invalid[signal] = int(
            (original.time_seconds >= case.opstart_seconds).sum()
            - (clean.time_seconds >= case.opstart_seconds).sum()
        )
        frame[signal], frame[f"{signal}_age_seconds"] = causal_hold(clean, seconds)
    map_values, sbp, dbp = frame["map"], frame.sbp, frame.dbp
    ordering_available = map_values.notna() & sbp.notna() & dbp.notna() & (sbp > 0) & (dbp > 0)
    frame["map_ordering_auditable"] = ordering_available
    frame["map_ordering_inconsistent"] = ordering_available & (
        (dbp > map_values) | (map_values > sbp)
    )
    audit = {
        **case.audit,
        "rejected_surgical_records": invalid,
        "map_ordering_auditable_seconds": int(ordering_available.sum()),
        "map_ordering_inconsistent_seconds": int(frame.map_ordering_inconsistent.sum()),
        "surgical_grid_seconds": len(frame),
        "valid_seconds": {signal: int(frame[signal].notna().sum()) for signal in TRACKS},
    }
    return ProcessedCase(
        case.case_id, case.subject_id, case.opstart_seconds, case.opend_seconds, frame, raw, audit
    )
