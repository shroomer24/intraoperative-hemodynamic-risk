"""Sealed model/protocol validation and held-out execution capabilities.

Approvals bind the prepared hashes. One-time reservations survive failures;
there is no automatic retry or model/configuration override path.
"""

import importlib.metadata
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from intraop.data.modeling import _load_partition, read_json
from intraop.evaluation.result_checkpoint import atomic_json, file_hash

MODEL_NAMES = (
    "prevalence",
    "current_map",
    "logistic_map",
    "logistic_full",
    "xgboost",
    "tabpfn_map",
    "tabpfn_full",
)
EXPECTED_IDS = dict(
    zip(
        MODEL_NAMES,
        (
            "fixed",
            "fixed",
            "C_10",
            "C_1",
            "n200_d2_lr0p05",
            "v3p5_default_seed42",
            "v3p5_default_seed42",
        ),
        strict=True,
    )
)
CALIBRATED_MODELS = ("logistic_map", "logistic_full", "xgboost", "tabpfn_map", "tabpfn_full")
CALIBRATION_PROTOCOL = {
    "method": "Platt_logit_logistic",
    "models": list(CALIBRATED_MODELS),
    "input": "logit(clip(raw_probability, 1e-6, 1-1e-6))",
    "epsilon": 1e-6,
    "parameters": {
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 2000,
        "tol": 0.0001,
        "fit_intercept": True,
        "class_weight": None,
        "random_state": 42,
    },
    "fit_partition": "calibration",
    "algorithm_comparison": False,
    "probability_outputs": ["raw", "calibrated"],
    "prevalence": "fixed training prevalence; no recalibration",
    "current_map": "-map_latest; AP/AUROC only; no probability transform",
    "model_configuration_changes_allowed": False,
    "reliability_bins": 10,
}
TEST_PROTOCOL = {
    "primary_metric": "average_precision",
    "secondary_metrics": ["auroc", "brier_score", "log_loss", "reliability_10_equal_width_bins"],
    "bootstrap_replicates": 1000,
    "bootstrap_unit": "subject",
    "bootstrap_seed": 42,
    "interval": [2.5, 97.5],
    "resampling": "subjects with replacement; all windows including multiplicity",
    "single_class_replicate": "AP and AUROC undefined; omit per metric and report valid counts",
    "current_map_metrics": ["average_precision", "auroc"],
    "raw_probabilities_primary": True,
    "calibrated_probabilities_secondary": True,
    "automatic_rerun": False,
    "optimization_after_lock": False,
}


def ensure_study_open(study):
    if (Path(study) / "study_closure_manifest.json").exists():
        raise PermissionError(
            "Optimization study CLOSED; B3/C/D and further optimization forbidden"
        )


def verify_seal(path, expected=None):
    path = Path(path)
    if not path.is_file() or not path.with_suffix(".sha256").is_file():
        raise PermissionError("Required sealed manifest absent")
    actual = file_hash(path)
    if path.with_suffix(".sha256").read_text().strip() != actual or (
        expected is not None and actual != expected
    ):
        raise PermissionError("Required manifest seal differs")
    return read_json(path), actual


def verify_files(root, hashes):
    for name, expected in hashes.items():
        path = Path(root) / name
        if not path.is_file() or file_hash(path) != expected:
            raise PermissionError("Bound file hash differs")


def validate_lock(modeling, *, expected=None, check_environment=False):
    modeling = Path(modeling)
    repo = modeling.parents[1]
    lock, lock_hash = verify_seal(modeling / "model_lock_manifest.json", expected)
    plan = read_json(modeling / "development_plan.json")
    completed = read_json(modeling / "development/development_completion_manifest.json")
    selected = read_json(modeling / "development/selected_configs.json")
    if lock["models"] != selected or tuple(lock["models"]) != MODEL_NAMES:
        raise PermissionError("Locked selected models differ from completed development")
    if lock["feature_sets"] != plan["feature_sets"]:
        raise PermissionError("Locked feature order differs")
    for name in MODEL_NAMES:
        config = lock["models"][name]["selected_configuration"]
        if config["candidate_id"] != EXPECTED_IDS[name]:
            raise PermissionError("Final approved candidate differs")
        definition = plan["models"][name]
        if config not in definition["candidates"] or any(
            lock["models"][name][key] != definition[key]
            for key in ("kind", "features", "preprocessing")
        ):
            raise PermissionError("Locked scientific definition differs")
    if lock["training_context"] != plan["training_context"]:
        raise PermissionError("Locked training context differs")
    if (
        lock["selection_metric"] != plan["selection_metric"]
        or lock["tie_break"] != plan["tie_break"]
    ):
        raise PermissionError("Selection protocol changed")
    if lock["tabpfn_representation"] != "A_frozen74_retained_control":
        raise PermissionError("Only baseline A is locked")
    if (
        lock["calibration_protocol"] != CALIBRATION_PROTOCOL
        or lock["test_protocol"] != TEST_PROTOCOL
    ):
        raise PermissionError("Locked held-out protocol changed")
    if lock["test_evaluations_before_lock"] != 0 or not lock["TEST_never_evaluated_before_lock"]:
        raise PermissionError("Pre-lock TEST evidence invalid")
    cohort = repo / lock["cohort_checkpoint"]
    verify_files(cohort, lock["dataset_manifest_hashes"])
    verify_files(cohort, lock["table_hashes"])
    verify_files(repo, lock["evidence_hashes"])
    verify_files(repo, lock["code_hashes"])
    if completed["test_evaluation_count"] != 0 or completed["calibration_accessed"]:
        raise PermissionError("Development held-out firewall violated")
    closure_path = repo / "artifacts/tabpfn-representation-study-v01/study_closure_manifest.json"
    closure, _ = verify_seal(closure_path, lock["optimization_study_closure_sha256"])
    if closure["status"] != "CLOSED" or closure["additional_optimization_permitted"]:
        raise PermissionError("Study is not closed")
    verify_files(closure_path.parent, closure["historical_attempt_files_sha256"])
    split = read_json(cohort / "realized_split_manifest.json")
    validate_subject_disjointness(split)
    if check_environment:
        if (
            sys.executable != lock["python_executable"]
            or sys.version.split()[0] != lock["python_version"]
        ):
            raise PermissionError("Pinned Python differs")
        actual = {n: importlib.metadata.version(n) for n in lock["package_versions"]}
        if actual != lock["package_versions"]:
            raise PermissionError("Pinned scientific packages differ")
    return lock, lock_hash


def validate_subject_disjointness(split):
    assignment = split["subject_to_partition"]
    if set(assignment.values()) != {"training", "tuning", "calibration", "test"}:
        raise PermissionError("Unexpected partition")
    rosters = {p: {s for s, v in assignment.items() if v == p} for p in split["counts"]}
    if any(len(rosters[p]) != split["counts"][p] for p in rosters):
        raise PermissionError("Subject counts differ")
    if sum(map(len, rosters.values())) != len(assignment):
        raise PermissionError("Subject partitions overlap")
    return rosters


def phase_paths(modeling, phase):
    root = Path(modeling) / "locked_execution" / phase
    return root, root / "execution_manifest.json", root / "completion_receipt.json"


def validate_approval(modeling, phase, lock_hash):
    approval = Path(modeling) / "locked_execution" / "approvals" / f"{phase}.json"
    if not approval.is_file():
        raise PermissionError("Architectural review approval required before held-out execution")
    value = read_json(approval)
    protocol_path = Path(modeling) / f"{phase}_protocol.json"
    _, protocol_hash = verify_seal(protocol_path)
    if value != {
        "status": "APPROVED",
        "phase": phase,
        "model_lock_sha256": lock_hash,
        "protocol_sha256": protocol_hash,
        "one_execution_only": True,
    }:
        raise PermissionError("Architectural approval does not bind exact lock/protocol")


def verify_calibration_complete(modeling, lock_hash):
    root, _, receipt = phase_paths(modeling, "calibration")
    value, _ = verify_seal(receipt)
    if value["status"] != "COMPLETE" or value["model_lock_sha256"] != lock_hash:
        raise PermissionError("Required locked calibration is incomplete")
    if value["models"] != list(MODEL_NAMES):
        raise PermissionError("Calibration models incomplete")
    verify_files(root, value["files_sha256"])
    return value


class LockedPartitions:
    """Only an approved, reserved external phase grants held-out value access."""

    def __init__(self, modeling, phase, reservation, *, check_environment=True):
        if phase not in ("calibration", "test"):
            raise PermissionError("Unknown execution phase")
        self.modeling, self.phase, self.reservation = Path(modeling), phase, reservation
        self.lock, self.lock_hash = validate_lock(modeling, check_environment=check_environment)
        self.repo = self.modeling.parents[1]
        self.cohort = self.repo / self.lock["cohort_checkpoint"]
        self.access_log = []

    def authorize(self):
        lock, lock_hash = validate_lock(self.modeling, expected=self.lock_hash)
        validate_approval(self.modeling, self.phase, lock_hash)
        root, manifest, receipt = phase_paths(self.modeling, self.phase)
        if receipt.exists() or (root / "failure.json").exists():
            raise PermissionError("Completed or failed phase cannot automatically repeat")
        if not manifest.is_file() or read_json(manifest) != self.reservation:
            raise PermissionError("Exact execution reservation is required")
        if self.reservation.get("status") != "RESERVED" or (
            self.reservation.get("model_lock_sha256") != lock_hash
        ):
            raise PermissionError("Invalid execution reservation")
        if self.phase == "test":
            verify_calibration_complete(self.modeling, lock_hash)
            prep, _ = verify_seal(self.modeling / "final_test_execution_manifest.json")
            if prep["model_lock_sha256"] != lock_hash or prep["protocol"] != lock["test_protocol"]:
                raise PermissionError("Final TEST execution manifest differs")
        return lock

    def load(self, partition):
        if partition not in ("training", self.phase):
            raise PermissionError("Partition excluded by locked phase capability")
        self.authorize()  # before even routing any table line
        split = read_json(self.cohort / "realized_split_manifest.json")
        data = _load_partition(self.cohort, split, self.lock["feature_sets"]["full"], partition)
        self.access_log.append(partition)
        return data


def reserve_phase(modeling, phase):
    lock, lock_hash = validate_lock(modeling, check_environment=True)
    validate_approval(modeling, phase, lock_hash)
    if phase == "test":
        verify_calibration_complete(modeling, lock_hash)
        prep, _ = verify_seal(Path(modeling) / "final_test_execution_manifest.json")
        if prep["model_lock_sha256"] != lock_hash:
            raise PermissionError("TEST preparation belongs to another lock")
    root, path, receipt = phase_paths(modeling, phase)
    if path.exists() or receipt.exists() or (root / "failure.json").exists():
        raise PermissionError("Prior execution retained; no automatic rerun")
    value = {
        "status": "RESERVED",
        "phase": phase,
        "model_lock_sha256": lock_hash,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "automatic_retry": False,
    }
    atomic_json(path, value)
    return lock, value


def bootstrap_subject_indices(groups, *, replicates=1000, seed=42):
    groups = np.asarray(groups)
    subjects = np.unique(groups)
    if not len(subjects):
        raise ValueError("No subjects")
    positions = [np.flatnonzero(groups == subject) for subject in subjects]
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        yield np.concatenate([positions[i] for i in rng.integers(0, len(subjects), len(subjects))])
