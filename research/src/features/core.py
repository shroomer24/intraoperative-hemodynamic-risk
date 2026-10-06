"""Stable 74-column causal numeric feature schema for protocol v0.1."""

import numpy as np
import pandas as pd

from intraop.data.protocol import HISTORY_SECONDS, MIN_SLOPE_OBSERVATIONS, TRACKS
from intraop.features.base import Window

SUMMARY_SUFFIXES = (
    "latest",
    "median_60s",
    "median_300s",
    "std_300s",
    "p10_300s",
    "p90_300s",
    "slope_60s",
    "slope_300s",
)
FEATURE_NAMES = (
    *(f"{signal}_{suffix}" for signal in TRACKS for suffix in SUMMARY_SUFFIXES),
    "map_median_latest_minus_previous_minute",
    *(f"map_minute_{minute}_median" for minute in range(1, 6)),
    "map_fraction_below_70",
    "map_fraction_below_75",
    "pulse_pressure_latest",
    "pulse_pressure_median_300s",
    "pulse_pressure_slope_300s",
    "hr_sbp_ratio_latest",
    "hr_sbp_ratio_median_60s",
    *(
        f"{signal}_{suffix}"
        for signal in TRACKS
        for suffix in ("missing_fraction", "measurement_age_seconds")
    ),
    "elapsed_seconds",
)


def _finite(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _median(values) -> float:
    values = _finite(values)
    return float(np.median(values)) if len(values) else np.nan


def _latest(values) -> float:
    values = _finite(values)
    return float(values[-1]) if len(values) else np.nan


def least_squares_slope(times, values) -> float:
    """OLS in value units per elapsed second; missing rows do not compress time."""
    times, values = np.asarray(times, dtype=float), np.asarray(values, dtype=float)
    mask = np.isfinite(times) & np.isfinite(values)
    times, values = times[mask], values[mask]
    if len(times) < MIN_SLOPE_OBSERVATIONS or len(np.unique(times)) < MIN_SLOPE_OBSERVATIONS:
        return np.nan
    centered = times - times.mean()
    return float(np.dot(centered, values - values.mean()) / np.dot(centered, centered))


class CoreFeatureExtractor:
    feature_names = FEATURE_NAMES

    def extract(self, window: Window) -> dict[str, float]:
        t = window.anchor_time_seconds
        history = window.history
        if t - window.history_start_seconds != HISTORY_SECONDS or len(history) != HISTORY_SECONDS:
            raise ValueError("Core features require the exact 300-second history")
        expected = np.arange(t - HISTORY_SECONDS + 1, t + 1, dtype=float)
        if not np.array_equal(history.time_seconds.to_numpy(), expected):
            raise ValueError("Core history must be the contiguous integer-second grid ending at t")
        recent_mask = history.time_seconds > t - 60
        previous_mask = (history.time_seconds > t - 120) & (history.time_seconds <= t - 60)
        result = {}
        for signal in TRACKS:
            full = history[signal].to_numpy(dtype=float)
            finite = _finite(full)
            raw = window.raw_samples.get(
                signal, pd.DataFrame(columns=["time_seconds", "value"], dtype=float)
            )
            raw = raw.drop_duplicates("time_seconds", keep="last")
            short = raw.loc[raw.time_seconds > t - 60]
            percentiles = np.percentile(finite, [10, 90]) if len(finite) else [np.nan, np.nan]
            summaries = (
                _latest(full),
                _median(full[recent_mask]),
                _median(full),
                float(np.std(finite, ddof=0)) if len(finite) else np.nan,
                *percentiles,
                least_squares_slope(short.time_seconds, short.value),
                least_squares_slope(raw.time_seconds, raw.value),
            )
            result.update(
                {
                    f"{signal}_{suffix}": float(value)
                    for suffix, value in zip(SUMMARY_SUFFIXES, summaries, strict=True)
                }
            )
        result["map_median_latest_minus_previous_minute"] = _median(
            history.loc[recent_mask, "map"]
        ) - _median(history.loc[previous_mask, "map"])
        for minute in range(1, 6):
            start = t - 300 + (minute - 1) * 60
            values = history.loc[
                (history.time_seconds > start) & (history.time_seconds <= start + 60), "map"
            ]
            result[f"map_minute_{minute}_median"] = _median(values)
        valid_map = _finite(history["map"])
        for threshold in (70, 75):
            result[f"map_fraction_below_{threshold}"] = (
                float(np.mean(valid_map < threshold)) if len(valid_map) else np.nan
            )
        pp = history.sbp - history.dbp
        result["pulse_pressure_latest"] = _latest(pp)
        result["pulse_pressure_median_300s"] = _median(pp)
        enough = all(
            window.raw_samples.get(
                signal, pd.DataFrame(columns=["time_seconds"])
            ).time_seconds.nunique()
            >= MIN_SLOPE_OBSERVATIONS
            for signal in ("sbp", "dbp")
        )
        result["pulse_pressure_slope_300s"] = (
            least_squares_slope(history.time_seconds, pp) if enough else np.nan
        )
        ratio = history.hr / history.sbp.where(history.sbp > 0)
        ratio = ratio.where(np.isfinite(ratio))
        result["hr_sbp_ratio_latest"] = _latest(ratio)
        result["hr_sbp_ratio_median_60s"] = _median(ratio[recent_mask])
        for signal in TRACKS:
            result[f"{signal}_missing_fraction"] = float(history[signal].isna().mean())
            # Latest actual recording observation age, including stale observations;
            # NaN if none has arrived by t. Staleness never makes its value current.
            result[f"{signal}_measurement_age_seconds"] = float(
                history[f"{signal}_age_seconds"].iloc[-1]
            )
        result["elapsed_seconds"] = float(t)
        if tuple(result) != FEATURE_NAMES:
            raise RuntimeError("Feature schema order changed")
        return result
