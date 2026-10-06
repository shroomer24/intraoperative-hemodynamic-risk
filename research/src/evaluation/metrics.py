"""Generic metrics; positive class and threshold must always be supplied."""

from collections.abc import Hashable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_classification_metrics(
    y_true: Sequence,
    positive_probabilities: Sequence[float],
    *,
    positive_label: Hashable,
    negative_label: Hashable,
    threshold: float,
) -> dict[str, float | None]:
    """Compute window-weighted binary metrics, without inferring clinical meaning.

    Supply the probability column matching positive_label in model.classes_.
    AUROC and average precision are None for single-class cohorts. Precision and
    recall are None when their denominator is zero. This does not provide patient
    weighting, confidence intervals, calibration curves, or alarm/event metrics.
    """
    if positive_label == negative_label:
        raise ValueError("Positive and negative labels must differ")
    truth = np.asarray(y_true)
    probabilities = np.asarray(positive_probabilities, dtype=float)
    if truth.ndim != 1 or truth.size == 0 or probabilities.shape != truth.shape:
        raise ValueError("Labels and probabilities must be aligned nonempty vectors")
    allowed = (truth == positive_label) | (truth == negative_label)
    if not allowed.all():
        raise ValueError("Labels must match the two explicitly supplied class values")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    if isinstance(threshold, bool) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    binary_truth = (truth == positive_label).astype(int)
    predictions = (probabilities >= threshold).astype(int)
    both_classes = np.unique(binary_truth).size == 2
    return {
        "accuracy": float(accuracy_score(binary_truth, predictions)),
        "precision": float(precision_score(binary_truth, predictions))
        if predictions.any()
        else None,
        "recall": float(recall_score(binary_truth, predictions)) if binary_truth.any() else None,
        "brier_score": float(brier_score_loss(binary_truth, probabilities)),
        "auroc": float(roc_auc_score(binary_truth, probabilities)) if both_classes else None,
        "average_precision": (
            float(average_precision_score(binary_truth, probabilities)) if both_classes else None
        ),
    }
