"""Execution-only continuation utilities; original scientific modules stay pinned."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from intraop.data.modeling import DevelopmentPartitions, read_json, sha256
from intraop.evaluation.secure_errors import sanitized_traceback

REPO = Path(__file__).resolve().parents[1]
PLAN = REPO / "artifacts/modeling-v01/development_plan.json"
REMEDIATION = REPO / "artifacts/modeling-v01/process_isolation_manifest.json"
FORBIDDEN = ("torch", "tabpfn", "tabpfn_client")


def assert_isolated():
    if any(name.split(".")[0] in FORBIDDEN for name in sys.modules):
        raise RuntimeError("XGBoost process isolation failure: forbidden runtime imported")


def original_runner():
    spec = importlib.util.spec_from_file_location(
        "original_development", REPO / "scripts/modeling_development.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verified_plan():
    manifest = read_json(REMEDIATION)
    if sha256(REMEDIATION) != REMEDIATION.with_suffix(".sha256").read_text().strip():
        raise ValueError("Execution isolation manifest hash mismatch")
    expected_code = manifest["code_hashes"]
    recovery = REPO / "artifacts/modeling-v01/attempt_recovery_manifest.json"
    if recovery.exists():
        if sha256(recovery) != recovery.with_suffix(".sha256").read_text().strip():
            raise ValueError("Attempt recovery manifest hash mismatch")
        revision = read_json(recovery)
        if revision["previous_execution_manifest_sha256"] != sha256(REMEDIATION):
            raise ValueError("Historical execution manifest changed")
        if revision["original_plan_sha256"] != manifest["original_plan_sha256"]:
            raise ValueError("Recovery changed scientific plan")
        for name, digest in manifest["code_hashes"].items():
            if sha256(REPO / revision["previous_implementation_directory"] / name) != digest:
                raise ValueError("Historical execution source evidence changed")
        for name, digest in revision["ledger_files_sha256"].items():
            if sha256(REPO / name) != digest:
                raise ValueError("Authorized attempt ledger changed")
        expected_code = revision["code_hashes"]
        manifest = {**manifest, "attempt_recovery_manifest_sha256": sha256(recovery)}
    for name, digest in expected_code.items():
        if sha256(REPO / name) != digest:
            raise ValueError(f"Execution isolation code changed: {name}")
    if sha256(PLAN) != manifest["original_plan_sha256"]:
        raise ValueError("Immutable development plan changed")
    plan = read_json(PLAN)
    original_runner().validate_environment(plan, require_token=False)
    for name, digest in plan["code_hashes"].items():
        if sha256(REPO / name) != digest:
            raise ValueError(f"Original scientific code changed: {name}")
    if (REPO / "artifacts/modeling-v01/model_lock_manifest.json").exists():
        raise PermissionError("Development forbidden after model lock")
    return plan, manifest


def load_training(plan):
    loader = DevelopmentPartitions(
        REPO / plan["cohort_checkpoint"],
        REPO / "artifacts/modeling-v01",
        plan["dataset_manifest_hashes"],
    )
    if loader.feature_sets != read_json(REPO / "artifacts/modeling-v01/feature_sets.json"):
        raise ValueError("Feature contract differs")
    if loader.features != plan["feature_sets"]["full"] or len(loader.features) != 74:
        raise ValueError("Frozen 74 predictors differ")
    if (
        sha256(REPO / "artifacts/modeling-v01/feature_sets.json")
        != plan["feature_sets_file_sha256"]
    ):
        raise ValueError("Feature contract hash differs")
    training = loader.load("training")
    if len(training.X) != 13975 or np.unique(training.y).tolist() != [0, 1]:
        raise ValueError("Frozen TRAINING count/classes differ")
    return loader, training


def candidate_definition(plan, model, candidate_id):
    definition = plan["models"][model]
    matches = [c for c in definition["candidates"] if c["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise ValueError("Candidate absent from immutable plan")
    return definition, matches[0]


def atomic_bytes(path, content, *, exclusive=False):
    """Publish complete bytes only; exclusive mode cannot replace a checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".atomic-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(path, value, *, exclusive=False):
    atomic_bytes(
        path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode(), exclusive=exclusive
    )


def sanitize_output(text):
    # Never serialize frame locals. Reuse the existing credential/payload scrubber.
    return sanitized_traceback(RuntimeError(text))


class ChildFailure(RuntimeError):
    def __init__(self, returncode):
        self.returncode = returncode
        self.signal = -returncode if returncode < 0 else None
        super().__init__(f"Candidate child failed: exit={returncode}, signal={self.signal}")


def supervise(command, log_dir, *, env=None):
    log_dir.mkdir(parents=True, exist_ok=False)
    result = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    atomic_bytes(log_dir / "stdout.txt", sanitize_output(result.stdout).encode(), exclusive=True)
    atomic_bytes(log_dir / "stderr.txt", sanitize_output(result.stderr).encode(), exclusive=True)
    evidence = {
        "command_entrypoint": Path(command[2]).name,
        "python_executable": command[0],
        "returncode": result.returncode,
        "signal": -result.returncode if result.returncode < 0 else None,
        "signal_name": signal.Signals(-result.returncode).name if result.returncode < 0 else None,
        "diagnostics_sanitized": True,
    }
    atomic_json(log_dir / "exit.json", evidence, exclusive=True)
    if result.returncode:
        atomic_json(
            log_dir / "failure_marker.json", {"status": "FAILED", **evidence}, exclusive=True
        )
        raise ChildFailure(result.returncode)
    return evidence


def validate_prediction(path, tuning, model, candidate_id):
    rows = pd.read_csv(path, dtype={"window_id": str, "subject_id": str, "case_id": str})
    if len(rows) != len(tuning.X):
        raise ValueError("Candidate prediction count differs")
    for column, expected in (
        ("source_row", tuning.X.index.to_numpy()),
        ("true_label", tuning.y.to_numpy()),
        ("window_id", tuning.metadata.window_id.to_numpy()),
    ):
        if not np.array_equal(rows[column].to_numpy(), expected):
            raise ValueError("Candidate prediction row alignment differs")
    for column, expected in (
        ("split", "tuning"),
        ("model_identifier", model),
        ("candidate_id", candidate_id),
    ):
        if set(rows[column]) != {expected}:
            raise ValueError("Candidate prediction identity differs")
    return rows
