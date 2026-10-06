"""Artificial software-semantic fixtures, never empirical clinical observations."""

import numpy as np
import pandas as pd

from intraop.data.preprocessing import ProcessedCase
from intraop.data.protocol import TRACKS
from intraop.features.base import Window


def processed(values, *, case_id="c1", subject_id="s1", opstart=0, opend=None):
    values = np.asarray(values, dtype=float)
    times = np.arange(len(values), dtype=float)
    if opend is None:
        opend = float(times[-1])
    mask = (times >= opstart) & (times <= opend)
    grid = pd.DataFrame({"time_seconds": times[mask]})
    raw = {}
    for signal in TRACKS:
        signal_values = values[mask] if signal == "map" else np.full(mask.sum(), np.nan)
        grid[signal] = signal_values
        grid[f"{signal}_age_seconds"] = np.where(np.isfinite(signal_values), 0.0, np.nan)
        good = np.isfinite(signal_values)
        raw[signal] = pd.DataFrame(
            {"time_seconds": times[mask][good], "value": signal_values[good]}
        )
    grid["map_ordering_inconsistent"] = False
    grid["map_ordering_auditable"] = False
    return ProcessedCase(case_id, subject_id, float(opstart), float(opend), grid, raw, {})


def window(case, anchor=300):
    history = case.grid.loc[
        (case.grid.time_seconds > anchor - 300) & (case.grid.time_seconds <= anchor)
    ].copy()
    raw = {
        s: data.loc[(data.time_seconds > anchor - 300) & (data.time_seconds <= anchor)].copy()
        for s, data in case.raw_valid_samples.items()
    }
    return Window(
        case.case_id,
        case.subject_id,
        float(anchor),
        float(anchor - 300),
        float(anchor),
        history,
        raw,
    )
