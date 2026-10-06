"""Splits return integer row positions, never DataFrame index labels."""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    test: np.ndarray


def _groups(groups: pd.Series) -> pd.Series:
    if groups.empty or groups.isna().any():
        raise ValueError("Provide nonempty, nonmissing patient groups")
    if any(not isinstance(g, str) or not g.strip() for g in groups):
        raise ValueError("Patient groups must be nonempty strings")
    return groups.reset_index(drop=True)


def patient_group_split(groups: pd.Series, *, test_fraction: float, seed: int) -> SplitIndices:
    """Seeded holdout by patient count; no class stratification is assumed.

    Sorting group IDs before shuffling makes group assignment independent of row
    order. The requested fraction applies to patients, not windows. This split
    alone imposes no chronological separation.
    """
    groups = _groups(groups)
    unique = sorted(groups.unique())
    if len(unique) < 2:
        raise ValueError("At least two patients are required")
    if isinstance(test_fraction, bool) or not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    shuffled = np.random.default_rng(seed).permutation(unique)
    count = min(len(unique) - 1, math.ceil(len(unique) * test_fraction))
    mask = groups.isin(shuffled[:count]).to_numpy()
    return SplitIndices(train=np.flatnonzero(~mask), test=np.flatnonzero(mask))


def grouped_temporal_split(
    groups: pd.Series,
    record_start: pd.Series,
    record_end: pd.Series,
    *,
    cutoff: pd.Timestamp,
) -> SplitIndices:
    """Earlier patients train; later patients test; crossing patients raise.

    Train requires each patient's latest record_end < cutoff. Test requires each
    patient's earliest record_start >= cutoff. Supply the COMPLETE dependency
    interval per row: input history plus any interval used to ascertain its label.
    Timezone conversion and horizon/embargo policy must be defined externally.
    Bare prediction anchors are insufficient for overlapping windows or labels.
    """
    groups_checked = _groups(groups)
    for times in (record_start, record_end):
        if not times.index.equals(groups.index):
            raise ValueError("Groups and interval timestamps must have identical indices")
        if not pd.api.types.is_datetime64_any_dtype(times) or times.isna().any():
            raise ValueError("Provide parsed, nonmissing datetime interval boundaries")
    if not isinstance(cutoff, pd.Timestamp) or pd.isna(cutoff):
        raise ValueError("cutoff must be an explicit pandas Timestamp")
    start = record_start.reset_index(drop=True)
    end = record_end.reset_index(drop=True)
    if (start > end).any():
        raise ValueError("Record start must not follow record end")
    intervals = pd.DataFrame({"group": groups_checked, "start": start, "end": end})
    bounds = intervals.groupby("group").agg(start=("start", "min"), end=("end", "max"))
    train_groups = bounds.index[bounds["end"] < cutoff]
    test_groups = bounds.index[bounds["start"] >= cutoff]
    if len(train_groups) + len(test_groups) != len(bounds):
        raise ValueError("Patient intervals cross the cutoff; define a cohort policy explicitly")
    if not len(train_groups) or not len(test_groups):
        raise ValueError("The cutoff must produce nonempty train and test cohorts")
    return SplitIndices(
        train=np.flatnonzero(groups_checked.isin(train_groups).to_numpy()),
        test=np.flatnonzero(groups_checked.isin(test_groups).to_numpy()),
    )
