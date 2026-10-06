"""Approved calibration or one-time TEST, with no scientific command-line overrides."""

from locked_bootstrap import REPO  # isort: skip

import argparse
import json
import os
import subprocess
import sys

import numpy as np

from intraop.data.modeling import read_json
from intraop.evaluation.calibration import apply_platt, bootstrap_intervals, fit_platt, reliability
from intraop.evaluation.development import development_metrics
from intraop.evaluation.locked_execution import (
    CALIBRATED_MODELS,
    MODEL_NAMES,
    LockedPartitions,
    phase_paths,
    reserve_phase,
    verify_calibration_complete,
)
from intraop.evaluation.result_checkpoint import atomic_json, file_hash, row_hash

MODELING = REPO / "artifacts/modeling-v01"


def seal_json(path, value):
    atomic_json(path, value)
    sidecar = path.with_suffix(".sha256")
    with sidecar.open("x") as stream:
        stream.write(file_hash(path) + "\n")
    sidecar.chmod(0o444)


def plot_reliability(root, tables):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    for name, bins in tables.items():
        occupied = [b for b in bins if b["windows"]]
        axis.plot(
            [b["mean_probability"] for b in occupied],
            [b["observed_frequency"] for b in occupied],
            marker="o",
            label=name,
        )
    axis.set(
        xlabel="Mean predicted probability",
        ylabel="Observed positive fraction",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(root / "reliability.png", dpi=160)
    plt.close(figure)


def execute(phase):
    if os.environ.get("INTRAOP_LOCKED_EXTERNAL_EXECUTION") != "normal_terminal":
        raise PermissionError("Use the approved secure normal-terminal wrapper")
    # Reservation persists even on failure. No retry is implied by absent receipt.
    lock, reservation = reserve_phase(MODELING, phase)
    root, _, receipt = phase_paths(MODELING, phase)
    try:
        for name in MODEL_NAMES:
            print(json.dumps({"status": "STARTING", "phase": phase, "model": name}), flush=True)
            result = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    str(REPO / "scripts/locked_model_worker.py"),
                    "--model",
                    name,
                    "--phase",
                    phase,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            atomic_json(
                root / "process_logs" / f"{name}.json",
                {
                    "exit_code": result.returncode,
                    "signal": -result.returncode if result.returncode < 0 else None,
                    "raw_streams_discarded": True,
                    "fresh_exec": True,
                    "model": name,
                    "model_lock_sha256": reservation["model_lock_sha256"],
                },
            )
            if result.returncode:
                raise RuntimeError("Locked worker failed; retained evidence requires review")
        loader = LockedPartitions(MODELING, phase, reservation)
        query = loader.load(phase)
        y = query.y.to_numpy(dtype=int)
        metrics, reliability_tables, mappings, bootstrap_scores = {}, {}, {}, {}
        previous = phase_paths(MODELING, "calibration")[0]
        if phase == "test":
            verify_calibration_complete(MODELING, loader.lock_hash)
            mappings = read_json(previous / "calibration_parameters.json")
        for name in MODEL_NAMES:
            worker = root / "workers" / name
            probability = name != "current_map"
            if probability:
                checkpoint = worker / "probability_checkpoint"
                manifest = read_json(checkpoint / "manifest.json")
                if (
                    manifest["contract"]["query_source_order_sha256"] != row_hash(query.X.index)
                    or manifest["contract"]["model_lock_sha256"] != loader.lock_hash
                    or file_hash(checkpoint / "probabilities.npy")
                    != manifest["probabilities_sha256"]
                ):
                    raise ValueError("Validated checkpoint alignment/hash differs")
                p = np.load(checkpoint / "probabilities.npy", allow_pickle=False)[:, 1]
            else:
                if read_json(worker / "score_contract.json")[
                    "query_source_order_sha256"
                ] != row_hash(query.X.index):
                    raise ValueError("Current-MAP alignment differs")
                p = np.load(worker / "scores.npy", allow_pickle=False)
            metrics[name + "/raw"] = development_metrics(y, p, probability_output=probability)
            bootstrap_scores[name + "/raw"] = p
            output = query.metadata.loc[
                :, ["window_id", "subject_id", "case_id", "anchor_time_seconds"]
            ].copy()
            output.insert(0, "source_row", query.X.index.to_numpy())
            output["true_label"] = y
            output["raw_probability"] = p if probability else np.nan
            output["raw_score"] = p if not probability else np.nan
            output["calibrated_probability"] = np.nan
            output["split"], output["model"] = phase, name
            output["model_lock_sha256"] = loader.lock_hash
            if probability:
                reliability_tables[name + "/raw"] = reliability(y, p)
            if name in CALIBRATED_MODELS:
                if phase == "calibration":
                    mappings[name] = fit_platt(p, y)
                calibrated = apply_platt(p, mappings[name])
                output["calibrated_probability"] = calibrated
                metrics[name + "/calibrated"] = development_metrics(
                    y, calibrated, probability_output=True
                )
                reliability_tables[name + "/calibrated"] = reliability(y, calibrated)
                bootstrap_scores[name + "/calibrated"] = calibrated
            if not output.index.equals(query.X.index) or output.window_id.duplicated().any():
                raise ValueError("Publication row alignment differs")
            path = root / "predictions" / f"{name}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            output.to_csv(path, index=False, mode="x")
            path.chmod(0o444)
        if phase == "calibration":
            atomic_json(root / "calibration_parameters.json", mappings)
        if phase == "test":
            atomic_json(
                root / "subject_bootstrap.json",
                bootstrap_intervals(y, query.groups, bootstrap_scores, replicates=1000, seed=42),
            )
        atomic_json(root / "metrics.json", metrics)
        atomic_json(root / "reliability.json", reliability_tables)
        plot_reliability(root, reliability_tables)
        loader.authorize()
        files = {
            str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*")) if p.is_file()
        }
        seal_json(
            receipt,
            {
                "status": "COMPLETE",
                "phase": phase,
                "models": list(MODEL_NAMES),
                "model_lock_sha256": loader.lock_hash,
                "rows": len(query.X),
                "subjects": query.groups.nunique(),
                "query_source_order_sha256": row_hash(query.X.index),
                "files_sha256": files,
                "automatic_rerun": False,
                "calibration_configuration_changes": False,
                "intervals_descriptive_subject_bootstrap": phase == "test",
                "calibration_metrics_are_in_sample_secondary": phase == "calibration",
            },
        )
        print(json.dumps({"status": "COMPLETE", "phase": phase}), flush=True)
    except BaseException as exc:
        atomic_json(
            root / "failure.json",
            {
                "status": "STOPPED_FOR_REVIEW",
                "exception_class": type(exc).__name__
                if type(exc).__name__
                in {"RuntimeError", "ValueError", "PermissionError", "FileExistsError"}
                else "OTHER",
                "model_lock_sha256": reservation["model_lock_sha256"],
                "raw_exception_messages_retained": False,
                "automatic_retry": False,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("calibration", "test"), required=True)
    args = parser.parse_args()
    try:
        execute(args.phase)
    except BaseException:
        print(
            "Locked execution blocked/stopped; inspect safe retained evidence. No automatic retry.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
