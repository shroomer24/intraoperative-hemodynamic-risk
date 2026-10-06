"""Single fresh public-replication model process. No sealed execution dependency."""

from locked_bootstrap import REPO  # isort: skip
import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO))
from intraop.models.benchmarks import (  # noqa: E402
    build_estimator,
    positive_probabilities,
    quiet_hosted_call,
)
from public_replication.cohort import SCOPE  # noqa: E402
from public_replication.runner import MODELS, save  # noqa: E402
from public_replication.safety import check_scope, install_io_boundary, owned_output  # noqa: E402


def run_model(name, payload, config):
    """Original fixed definitions; no tuning, calibration or diagnostic metrics."""
    features = config["feature_sets"]["full"]
    row = config["models"][name]
    chosen = [features.index(f) for f in row["features"]]
    train = payload["train_X"][:, chosen]
    query = payload["query_X"][:, chosen]
    if name == "current_map":
        return -query[:, 0], {"output_kind": "raw_ranking_score", "probability_validation": None}
    if name == "prevalence":
        train = np.zeros((len(train), 1))
        query = np.zeros((len(query), 1))
    with quiet_hosted_call():
        model = build_estimator(row["kind"], row["selected_configuration"]["hyperparameters"])
        model.fit(train, payload["train_y"])
        probabilities = positive_probabilities(model, query, require_ordered_classes=True)
    return probabilities, {
        "output_kind": "raw_probability",
        "classes": [0, 1],
        "positive_class_column": 1,
        "probability_validation": "PASS",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args()
    try:
        output = owned_output(args.run, existing=True)
        check_scope(output)
        target = output / "models" / args.model
        if list(target.iterdir()) not in [[], [target / "process.log"]]:
            raise PermissionError("Worker artifacts already exist; no duplicate execution")
        install_io_boundary(output)
        if args.model.startswith("tabpfn"):
            if importlib.metadata.version("tabpfn-client") != "0.6.1":
                raise ValueError("Exact tabpfn-client 0.6.1 required")
            if not os.environ.get("TABPFN_TOKEN"):
                raise PermissionError("Authenticated hosted access required")
        config = json.loads((REPO / "reference/selected_models.json").read_text())
        with np.load(output / "public_model_inputs.npz", allow_pickle=False) as payload:
            scores, validation = run_model(args.model, payload, config)
            rows = payload["query_rows"]
            if scores.shape != rows.shape or not np.isfinite(scores).all():
                raise ValueError("Query alignment or finite score validation failed")
            np.savez(target / "public_raw_predictions.npz", source_rows=rows, raw_output=scores)
        save(
            target / "completion.json",
            {
                "scope": SCOPE,
                "status": "PASS",
                "model": args.model,
                "query_rows": len(rows),
                "query_order_preserved": True,
                "calibration_applied": False,
                "metrics_computed": False,
                **validation,
            },
        )
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"scope": SCOPE, "status": "BLOCKED", "exception_class": type(exc).__name__}
            ),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
