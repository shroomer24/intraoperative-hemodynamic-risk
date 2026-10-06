"""Exact v0.1 sustained-MAP endpoint, eligibility, and symmetric future censoring."""

import math

import numpy as np
import pandas as pd

from intraop.data.preprocessing import ProcessedCase
from intraop.data.protocol import (
    ANCHOR_STRIDE_SECONDS,
    CONFIRMATION_SECONDS,
    FUTURE_SECONDS,
    HISTORY_SECONDS,
    HORIZON_SECONDS,
    MAP_THRESHOLD,
    MIN_HISTORY_COVERAGE,
)

EPISODE_COLUMNS = (
    "case_id",
    "subject_id",
    "episode_number",
    "onset_time_seconds",
    "confirmation_time_seconds",
    "is_recurrent",
    "evaluable",
)
ANCHOR_COLUMNS = (
    "case_id",
    "subject_id",
    "anchor_time_seconds",
    "history_start_seconds",
    "history_end_seconds",
    "future_observation_end_seconds",
    "history_map_coverage",
    "recent_map_valid_seconds",
    "future_map_valid_seconds",
    "future_required_seconds",
    "map_ordering_inconsistent_history_seconds",
    "status",
    "reason",
    "label",
    "matched_episode_onset_seconds",
    "matched_episode_is_recurrent",
)


def detect_episodes(case: ProcessedCase) -> pd.DataFrame:
    """First episode needs a confirmed run; recurrent episodes need recovery.

    Missing data resets low and recovery counters but does not rearm a previously
    confirmed episode. Episode state resets at every operation boundary.
    """
    low_run = recovery = 0
    armed = True
    rows = []
    for time, value in zip(case.grid.time_seconds, case.grid["map"], strict=True):
        if not np.isfinite(value):
            low_run = recovery = 0
        elif value >= MAP_THRESHOLD:
            low_run = 0
            recovery += 1
            if recovery >= CONFIRMATION_SECONDS:
                armed = True
        else:
            recovery = 0
            low_run += 1
            if armed and low_run == CONFIRMATION_SECONDS:
                rows.append(
                    {
                        "case_id": case.case_id,
                        "subject_id": case.subject_id,
                        "episode_number": len(rows) + 1,
                        "onset_time_seconds": float(time - CONFIRMATION_SECONDS + 1),
                        "confirmation_time_seconds": float(time),
                        "is_recurrent": bool(rows),
                        "evaluable": False,
                    }
                )
                armed = False
    return pd.DataFrame(rows, columns=EPISODE_COLUMNS)


def matched_episode(episodes: pd.DataFrame, anchor: float) -> pd.Series | None:
    matches = episodes.loc[
        (episodes.onset_time_seconds > anchor)
        & (episodes.onset_time_seconds <= anchor + HORIZON_SECONDS)
    ]
    return None if matches.empty else matches.iloc[0]


def label_anchors(case: ProcessedCase, episodes: pd.DataFrame) -> pd.DataFrame:
    """Cadence aligned to recording start; history lies completely within surgery.

    Full history (t-300,t] has exactly 300 seconds; recent includes t-59..t.
    Every labeled row requires all 360 seconds in (t,t+360] to be observable.
    Censoring is reported only for candidates that passed causal eligibility.
    """
    frame = case.grid.set_index("time_seconds", drop=False)
    first = math.ceil((case.opstart_seconds + HISTORY_SECONDS) / ANCHOR_STRIDE_SECONDS)
    anchors = range(
        first * ANCHOR_STRIDE_SECONDS, math.floor(case.opend_seconds) + 1, ANCHOR_STRIDE_SECONDS
    )
    rows = []
    for anchor in anchors:
        history = frame.loc[(frame.index > anchor - HISTORY_SECONDS) & (frame.index <= anchor)]
        recent = history.loc[history.index > anchor - CONFIRMATION_SECONDS, "map"]
        future = frame.loc[(frame.index > anchor) & (frame.index <= anchor + FUTURE_SECONDS), "map"]
        coverage = history["map"].notna().sum() / HISTORY_SECONDS
        reason = ""
        status = "eligible"
        if len(history) != HISTORY_SECONDS or coverage < MIN_HISTORY_COVERAGE:
            status, reason = "ineligible", "insufficient_history_coverage"
        elif (
            len(recent) != CONFIRMATION_SECONDS
            or not (recent.notna() & (recent >= MAP_THRESHOLD)).all()
        ):
            status, reason = "ineligible", "not_observably_nonhypotensive_for_60_seconds"
        elif anchor + FUTURE_SECONDS > case.opend_seconds:
            status, reason = "censored", "insufficient_remaining_surgical_observation"
        elif len(future) != FUTURE_SECONDS or not future.notna().all():
            status, reason = "censored", "missing_future_map"
        match = matched_episode(episodes, anchor) if status == "eligible" else None
        rows.append(
            {
                "case_id": case.case_id,
                "subject_id": case.subject_id,
                "anchor_time_seconds": float(anchor),
                "history_start_seconds": float(anchor - HISTORY_SECONDS),
                "history_end_seconds": float(anchor),
                "future_observation_end_seconds": float(anchor + FUTURE_SECONDS),
                "history_map_coverage": float(coverage),
                "recent_map_valid_seconds": int(recent.notna().sum()),
                "future_map_valid_seconds": int(future.notna().sum()),
                "future_required_seconds": FUTURE_SECONDS,
                "map_ordering_inconsistent_history_seconds": int(
                    history.map_ordering_inconsistent.sum()
                ),
                "status": status,
                "reason": reason,
                "label": int(match is not None) if status == "eligible" else np.nan,
                "matched_episode_onset_seconds": float(match.onset_time_seconds)
                if match is not None
                else np.nan,
                "matched_episode_is_recurrent": bool(match.is_recurrent)
                if match is not None
                else None,
            }
        )
    return pd.DataFrame(rows, columns=ANCHOR_COLUMNS)


def mark_evaluable_episodes(episodes: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    result = episodes.copy()
    eligible = anchors.loc[anchors.status == "eligible", "anchor_time_seconds"]
    for index, episode in episodes.iterrows():
        result.loc[index, "evaluable"] = bool(
            (
                (eligible < episode.onset_time_seconds)
                & (eligible >= episode.onset_time_seconds - HORIZON_SECONDS)
            ).any()
        )
    return result
