"""Development metrics, prespecified selection, and positional prediction export."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from intraop.data.datasets import ClassificationDataset


def development_metrics(y, scores, *, probability_output: bool) -> dict:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if scores.shape != y.shape or not np.isfinite(scores).all():
        raise ValueError("Scores and labels must be aligned and finite")
    if np.unique(y).tolist() != [0, 1]:
        raise ValueError("Development metrics require both frozen label classes")
    if probability_output and ((scores < 0).any() or (scores > 1).any()):
        raise ValueError("Probabilities must be in [0, 1]")
    return {
        "windows": len(y),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, scores)),
        "auroc": float(roc_auc_score(y, scores)),
        "log_loss": float(log_loss(y, scores, labels=[0, 1])) if probability_output else None,
        "brier_score": float(brier_score_loss(y, scores)) if probability_output else None,
        "probability_output": probability_output,
    }


def select_candidate(records: list[dict]) -> dict:
    # Simplicity ranks are assigned in the plan before fitting, never after results.
    return min(
        records,
        key=lambda row: (
            -row["average_precision"],
            row["log_loss"] if row["log_loss"] is not None else float("inf"),
            row["simplicity_rank"],
            row["candidate_id"],
        ),
    )


def export_predictions(
    data: ClassificationDataset, scores, *, model: str, candidate_id: str, probability_output: bool
) -> pd.DataFrame:
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(data.X),) or not np.isfinite(scores).all():
        raise ValueError("Export scores must match original row count/order")
    output = data.metadata.loc[
        :, ["window_id", "subject_id", "case_id", "anchor_time_seconds"]
    ].copy()
    output.insert(0, "source_row", data.X.index.to_numpy())
    output["true_label"] = data.y.to_numpy(dtype=int)
    output["predicted_probability"] = scores if probability_output else np.nan
    output["raw_score"] = scores if not probability_output else np.nan
    output["split"] = "tuning"
    output["model_identifier"] = model
    output["candidate_id"] = candidate_id
    if not output.index.equals(data.X.index):
        raise ValueError("Export index order differs")
    if output.window_id.duplicated().any():
        raise ValueError("Duplicate prediction window identifiers")
    return output
