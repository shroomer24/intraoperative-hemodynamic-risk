"""One prespecified calibration mapping, with raw results retained."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from intraop.evaluation.locked_execution import CALIBRATION_PROTOCOL, bootstrap_subject_indices


def logit_input(probabilities):
    epsilon = CALIBRATION_PROTOCOL["epsilon"]
    p = np.clip(np.asarray(probabilities, dtype=float), epsilon, 1 - epsilon)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_platt(probabilities, y):
    if np.unique(y).tolist() != [0, 1]:
        raise ValueError("Prespecified calibration requires both frozen label classes")
    model = LogisticRegression(**CALIBRATION_PROTOCOL["parameters"])
    model.fit(logit_input(probabilities), np.asarray(y, dtype=int))
    return {
        "method": "Platt_logit_logistic",
        "coefficient": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
        "epsilon": CALIBRATION_PROTOCOL["epsilon"],
        "fit_partition": "calibration",
        "configuration_changed": False,
    }


def apply_platt(probabilities, parameters):
    if parameters["method"] != "Platt_logit_logistic" or (
        parameters["epsilon"] != CALIBRATION_PROTOCOL["epsilon"]
    ):
        raise ValueError("Calibration mapping differs from locked protocol")
    z = parameters["coefficient"] * logit_input(probabilities)[:, 0] + parameters["intercept"]
    # Numerically stable sigmoid; no mapping/threshold optimization.
    from scipy.special import expit

    return expit(z)


def reliability(y, probabilities):
    y, p = np.asarray(y, dtype=int), np.asarray(probabilities, dtype=float)
    bins = np.minimum((p * 10).astype(int), 9)
    return [
        {
            "lower": i / 10,
            "upper": (i + 1) / 10,
            "windows": int((bins == i).sum()),
            "mean_probability": float(p[bins == i].mean()) if (bins == i).any() else None,
            "observed_frequency": float(y[bins == i].mean()) if (bins == i).any() else None,
        }
        for i in range(10)
    ]


def bootstrap_intervals(y, groups, scores, *, replicates=1000, seed=42):
    y = np.asarray(y, dtype=int)
    values = {name: {"average_precision": [], "auroc": []} for name in scores}
    for index in bootstrap_subject_indices(groups, replicates=replicates, seed=seed):
        target = y[index]
        if np.unique(target).size < 2:
            continue
        for name, p in scores.items():
            s = np.asarray(p)[index]
            values[name]["average_precision"].append(float(average_precision_score(target, s)))
            values[name]["auroc"].append(float(roc_auc_score(target, s)))
    return {
        name: {
            metric: {
                "valid_replicates": len(v),
                "undefined_replicates": replicates - len(v),
                "percentile_95": np.percentile(v, [2.5, 97.5]).tolist() if v else None,
            }
            for metric, v in row.items()
        }
        for name, row in values.items()
    }
