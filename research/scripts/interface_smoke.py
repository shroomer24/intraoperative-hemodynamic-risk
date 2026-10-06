"""Synthetic conventional-model smoke in a single fresh process; no clinical inputs."""

from locked_bootstrap import REPO  # isort: skip
import argparse
import json

import numpy as np

from intraop.models.benchmarks import build_estimator, positive_probabilities

p = argparse.ArgumentParser(description=__doc__)
p.add_argument(
    "--model", choices=["prevalence", "logistic_map", "logistic_full", "xgboost"], required=True
)
a = p.parse_args()
d = json.loads((REPO / "reference/selected_models.json").read_text())["models"][a.model]
width = len(d["features"]) or 1
X = np.random.default_rng(42).normal(size=(64, width))
y = np.tile([0, 1], 32)
m = build_estimator(d["kind"], d["selected_configuration"]["hyperparameters"])
m.fit(X, y)
prob = positive_probabilities(m, X[:8], require_ordered_classes=True)
assert prob.shape == (8,)
print(
    json.dumps(
        {
            "model": a.model,
            "status": "PASS",
            "synthetic_only": True,
            "query_rows": 8,
            "classes": [0, 1],
        }
    )
)
