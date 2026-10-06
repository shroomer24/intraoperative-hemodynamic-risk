"""Approved MAP-only calibration recovery; old locked sources stay unchanged."""

import contextlib
import logging
import os
import shlex
import subprocess
import sys
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
MODELING = BASE.parent
REPO = MODELING.parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import locked_bootstrap  # noqa: E402,F401

from intraop.data.modeling import _load_partition, predictor_array, read_json  # noqa: E402
from intraop.evaluation.locked_execution import validate_lock, verify_seal  # noqa: E402
from intraop.evaluation.result_checkpoint import (  # noqa: E402
    CONSTRUCTOR,
    SEMANTICS,
    ScientificFailure,
    array_hash,
    atomic_json,
    digest,
    file_hash,
    optional_metadata,
    row_hash,
    validate_probabilities,
)

PYTHON = "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
LOCK_HASH = "b2718a7c1ba448cc319c8594f93dc022480c573ef3dec3babd9b14dbdd021c0d"
PROTOCOL_HASH = "4d4f948b6d4f534eab52c0b997fcb9e9006abb64114ac2f5b5272d39be238f6e"
PROBE_HASH = "9ace0de73168dbdfbca9dc268e7994ae3080513156735344c1c4d6f3608776c4"
DRAFT_HASH = "b37e9a1a7043b4526cda261c93655fa69955279394e1fa8ae273be322aa8c465"
WORKER_FAILURE_HASH = "050214cc4c9328748354c21cc4fbb2804d5ce3470b5b05c72639da49548fb26c"
SUPERVISOR_FAILURE_HASH = "6ea8f52b0d26cb055475af6cf3b7927e8f0546f8e6573e0f96e5e6e370e45c2b"
QUERY_ORDER_HASH = "654647bd639559261b2ac23b66376c96ff9c583a86145750566f2412b12a2004"
FEATURE_HASH = "32ca198f264e92caa36d101fc83d1f9efc72e3bee3ca4b400be0c12d4716ab26"
CANONICAL = "/app/tabpfn_models/tabpfn-v3.5-20260909.safetensors"
CALIBRATION = MODELING / "locked_execution/calibration"
ATTEMPT = CALIBRATION / "attempts/tabpfn_map/attempt_2"
AMENDMENT = BASE / "execution_amendment.json"
APPROVAL = MODELING / "locked_execution/approvals/calibration_map_attempt_2.json"
PRESERVED = ("prevalence", "current_map", "logistic_map", "logistic_full", "xgboost")


def stamp():
    return datetime.now(UTC).isoformat()


def seal_json(path, value):
    atomic_json(path, value)
    atomic_json_hash(path)


def atomic_json_hash(path):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".seal-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(file_hash(path) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        target = path.with_suffix(".sha256")
        os.link(temporary, target)
        target.chmod(0o444)
    finally:
        Path(temporary).unlink(missing_ok=True)


def resolve_attempt(model="tabpfn_map", number=2):
    if model != "tabpfn_map" or type(number) is not int or number != 2:
        raise PermissionError("Only approved MAP Attempt 2 is available")
    return ATTEMPT


def checked_state(*, external=False):
    amendment, amendment_hash = verify_seal(AMENDMENT)
    if (
        amendment["scope"] != {"phase": "calibration", "model": "tabpfn_map", "attempt": 2}
        or amendment["model_lock_sha256"] != LOCK_HASH
        or amendment["calibration_protocol_sha256"] != PROTOCOL_HASH
        or amendment["synthetic_probe_sha256"] != PROBE_HASH
        or amendment["reviewed_draft_sha256"] != DRAFT_HASH
        or amendment["development_identity"] != "DEVELOPMENT_CONCRETE_IDENTITY_UNAVAILABLE"
        or amendment["checkpoint_identical_development_continuity_claimed"]
        or amendment["full_or_test_execution_authorized"]
    ):
        raise PermissionError("Approved recovery scope differs")
    approval, _ = verify_seal(APPROVAL)
    if approval != {
        "status": "APPROVED",
        "scope": {"phase": "calibration", "model": "tabpfn_map", "attempt": 2},
        "execution_amendment_sha256": amendment_hash,
        "model_lock_sha256": LOCK_HASH,
        "calibration_protocol_sha256": PROTOCOL_HASH,
        "synthetic_probe_sha256": PROBE_HASH,
        "reviewed_draft_sha256": DRAFT_HASH,
        "approval_source_sha256": amendment["approval_source_sha256"],
        "maximum_fit_calls": 1,
        "maximum_predict_proba_calls": 1,
        "automatic_retry": False,
        "tabpfn_full_authorized": False,
        "TEST_authorized": False,
    }:
        raise PermissionError("Specific architectural approval differs")
    for relative, expected in amendment["bound_files_sha256"].items():
        if file_hash(REPO / relative) != expected:
            raise PermissionError("Bound recovery/provenance source differs")
    for relative, expected in amendment["protected_inventory"].items():
        path = REPO / relative
        if (
            file_hash(path) != expected["sha256"]
            or path.stat().st_size != expected["bytes"]
            or oct(path.stat().st_mode & 0o777) != expected["mode"]
        ):
            raise PermissionError("Preserved scientific/worker evidence differs")
    for relative, expected in {
        "workers/tabpfn_map/failure.json": WORKER_FAILURE_HASH,
        "failure.json": SUPERVISOR_FAILURE_HASH,
    }.items():
        if file_hash(CALIBRATION / relative) != expected:
            raise PermissionError("Only the exact reviewed historical failure is allowed")
    # Any new file outside the single new attempt is unreviewed execution state.
    protected = set(amendment["protected_inventory"])
    for path in CALIBRATION.rglob("*"):
        if path.is_file() and not path.is_relative_to(ATTEMPT):
            if str(path.relative_to(REPO)) not in protected:
                raise PermissionError("Unreviewed calibration execution state")
    if (CALIBRATION / "workers/tabpfn_full").exists():
        raise PermissionError("Full TabPFN is not authorized by this capability")
    execution = MODELING / "locked_execution"
    if (execution / "test").exists() or (execution / "approvals/test.json").exists():
        raise PermissionError("TEST unavailable during MAP-only review boundary")
    parent = ATTEMPT.parent
    if parent.exists() and any(path.name != "attempt_2" for path in parent.iterdir()):
        raise PermissionError("No additional MAP attempt is permitted")
    for path in [CALIBRATION, parent.parent, parent, ATTEMPT]:
        if path.is_symlink():
            raise PermissionError("Recovery execution paths must not be symlinks")
    lock, actual = validate_lock(MODELING, expected=LOCK_HASH, check_environment=True)
    _, protocol = verify_seal(MODELING / "calibration_protocol.json", PROTOCOL_HASH)
    if actual != LOCK_HASH or protocol != PROTOCOL_HASH:
        raise PermissionError("Scientific seal differs")
    definition = lock["models"]["tabpfn_map"]
    if (
        definition["selected_configuration"]["hyperparameters"] != CONSTRUCTOR
        or definition["features"] != lock["feature_sets"]["map_only"]
        or len(definition["features"]) != 18
        or digest(definition["features"]) != FEATURE_HASH
    ):
        raise PermissionError("Only the frozen MAP configuration is available")
    if external and (
        sys.executable != PYTHON
        or os.environ.get("INTRAOP_MAP_RECOVERY_EXECUTION") != "normal_terminal"
        or not os.environ.get("TABPFN_TOKEN")
    ):
        raise PermissionError("Exact external interpreter/credential environment required")
    return amendment, amendment_hash, lock


def inspect_processes():
    result = subprocess.run(
        ["/bin/ps", "-axo", "pid=,command="], capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise PermissionError("Worker inactivity cannot be verified")
    active = []
    names = {
        "map_worker.py",
        "recover_map.py",
        "locked_model_worker.py",
        "run_locked_evaluation.py",
        "modeling_tabpfn_worker.py",
        "modeling_development_resume.py",
        "recovery_worker.py",
        "study_worker.py",
    }
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit() or int(parts[0]) == os.getpid():
            continue
        try:
            argv = shlex.split(parts[1])
        except ValueError:
            continue
        if any(Path(arg).name in names for arg in argv):
            active.append(int(parts[0]))
    if active:
        raise PermissionError("Relevant worker is active; no new request permitted")
    # Raw commands are never retained, including unrelated credential-bearing commands.
    return {
        "status": "NO_RELEVANT_WORKER_ACTIVE",
        "timestamp_utc": stamp(),
        "raw_commands_retained": False,
    }


def reserve_attempt(amendment_hash, process_check):
    path = resolve_attempt()
    if path.exists():
        raise FileExistsError("Prior Attempt 2 is retained; no duplicate or retry")
    if process_check.get("status") != "NO_RELEVANT_WORKER_ACTIVE":
        raise PermissionError("Live worker inactivity is required")
    path.mkdir(parents=True, exist_ok=False)
    reservation = {
        "status": "RESERVED",
        "phase": "calibration",
        "model": "tabpfn_map",
        "attempt": 2,
        "timestamp_utc": stamp(),
        "execution_amendment_sha256": amendment_hash,
        "model_lock_sha256": LOCK_HASH,
        "maximum_fit_calls": 1,
        "maximum_predict_proba_calls": 1,
        "automatic_retry": False,
    }
    atomic_json(path / "reservation.json", reservation)
    atomic_json(path / "process_inspection.json", process_check)
    return reservation


class RecoveryPartitions:
    """This approval grants TRAINING/CALIBRATION routing only; TEST has no route."""

    def __init__(self):
        _, self.amendment_hash, self.lock = checked_state(external=True)
        self.cohort = REPO / self.lock["cohort_checkpoint"]
        self.access_log = []

    def authorize(self):
        _, current, lock = checked_state(external=True)
        if current != self.amendment_hash:
            raise PermissionError("Recovery amendment changed")
        reservation = read_json(ATTEMPT / "reservation.json")
        if (
            reservation["status"] != "RESERVED"
            or reservation["execution_amendment_sha256"] != current
            or reservation["model"] != "tabpfn_map"
            or reservation["attempt"] != 2
            or (ATTEMPT / "failure.json").exists()
            or (ATTEMPT / "completion_receipt.json").exists()
        ):
            raise PermissionError("New failed/complete/indeterminate attempt cannot repeat")
        return lock

    def load(self, partition):
        if partition not in ("training", "calibration"):
            raise PermissionError("Partition unavailable to MAP recovery")
        self.authorize()  # before routing any table lines
        split = read_json(self.cohort / "realized_split_manifest.json")
        data = _load_partition(self.cohort, split, self.lock["feature_sets"]["full"], partition)
        self.access_log.append(partition)
        return data


def input_contract(loader, training, query):
    context = loader.lock["training_context"]
    features = loader.lock["models"]["tabpfn_map"]["features"]
    if (
        len(training.X) != context["rows"]
        or training.groups.nunique() != context["subjects"]
        or int(training.y.sum()) != context["positive_labels"]
        or np.unique(training.y).tolist() != [0, 1]
        or set(training.groups) & set(query.groups)
        or len(query.X) != 2393
        or row_hash(query.X.index) != QUERY_ORDER_HASH
        or len(features) != 18
        or digest(features) != FEATURE_HASH
    ):
        raise ScientificFailure("input_contract", None, "FROZEN_CONTEXT_MISMATCH")
    X, query_X = predictor_array(training, features), predictor_array(query, features)
    contract = {
        "candidate": "tabpfn_map",
        "kind": "tabpfn",
        "constructor": CONSTRUCTOR.copy(),
        "features": features,
        "feature_count": 18,
        "feature_contract_sha256": digest(features),
        "training_X_sha256": array_hash(X),
        "training_y_sha256": array_hash(training.y),
        "query_X_sha256": array_hash(query_X),
        "query_rows": len(query.X),
        "query_source_order_sha256": row_hash(query.X.index),
        "query_order_verified": bool(
            query.X.index.is_unique
            and query.X.index.is_monotonic_increasing
            and query.y.index.equals(query.X.index)
            and query.metadata.index.equals(query.X.index)
        ),
        "metadata_excluded_from_X": not bool(set(features) & set(query.metadata)),
        "model_lock_sha256": LOCK_HASH,
        "dataset_hashes": loader.lock["table_hashes"],
        "split_manifest_sha256": loader.lock["split_manifest_hash"],
        "software_versions": loader.lock["package_versions"],
        "positive_class": 1,
        "probability_column": 1,
        "positive_class_semantics": SEMANTICS,
    }
    return X, query_X, contract


def compatible_metadata(raw, *, rows, features):
    def fail(field, value, reason="CONTRACT_MISMATCH"):
        raise ScientificFailure(field, value, reason, stage="REQUIRED_METADATA_VALIDATION")

    if not isinstance(raw, dict):
        fail("metadata", raw, "INVALID_CONTAINER")
    config = raw.get("tabpfn_config")
    if not isinstance(config, dict):
        fail("tabpfn_config", config, "INVALID_CONTAINER")
    actual = config.get("model_path")
    if type(actual) is not str or actual not in ("v3.5_default", CANONICAL):
        fail("model_path", actual)
    for field, expected in [("billing_model_version", "v3.5"), ("execution_mode", "standard")]:
        value = raw.get(field)
        if type(value) is not str or value != expected:
            fail(field, value)
    required = {"model_path": actual, "billing_model_version": "v3.5", "execution_mode": "standard"}
    # Preserve the original mandatory dimension/class and non-identity configuration checks.
    for field, expected in [
        ("test_set_num_rows", rows),
        ("test_set_num_cols", features),
        ("classes", [0, 1]),
    ]:
        value = raw.get(field)
        if value is None:
            required[field] = "UNKNOWN"
            continue
        if field == "classes":
            valid = isinstance(value, (list, tuple, np.ndarray))
            if valid:
                a = np.asarray(value)
                valid = a.shape == (2,) and a.tolist() in ([0, 1], ["0", "1"])
        else:
            valid = type(value) is int and value == expected
        if not valid:
            fail(field, value)
        required[field] = expected
    for field, expected in [
        ("random_state", 42),
        ("balance_probabilities", False),
        ("ignore_pretraining_limits", False),
    ]:
        value = config.get(field)
        if value is not None and (type(value) is not type(expected) or value != expected):
            fail(field, value)
        required[field] = expected if value is not None else "UNKNOWN"
    return required


def persist_approved_result(
    path, probabilities, classes, contract, observed, raw, *, amendment_hash, mark
):
    mark("PROBABILITY_VALIDATION_STARTED")
    a = np.asarray(classes)
    if a.dtype.kind not in "iu" or a.shape != (2,) or a.tolist() != [0, 1]:
        raise ScientificFailure("classes", None, "ORDERED_BINARY_CLASSES_REQUIRED")
    if (
        contract.get("candidate") != "tabpfn_map"
        or contract.get("kind") != "tabpfn"
        or contract.get("model_lock_sha256") != LOCK_HASH
        or contract.get("feature_count") != 18
        or digest(contract.get("features")) != FEATURE_HASH
        or contract.get("feature_contract_sha256") != FEATURE_HASH
    ):
        raise ScientificFailure("input_contract", None, "MAP_ONLY_RECOVERY_SCOPE_MISMATCH")
    p = validate_probabilities(probabilities, classes, contract, observed)
    required = compatible_metadata(raw, rows=len(p), features=contract["feature_count"])
    mark("PROBABILITY_VALIDATION_PASSED")
    path = Path(path)
    if path.exists():
        raise FileExistsError("Immutable probability checkpoint exists")
    atomic_json(
        path.parent / (path.name + ".reservation.json"),
        {
            "status": "RESERVED",
            "scientific_contract_sha256": digest(contract),
            "execution_amendment_sha256": amendment_hash,
        },
    )
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=path.parent))
    with (temporary / "probabilities.npy").open("xb") as stream:
        np.save(stream, p, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    seal_json(
        temporary / "manifest.json",
        {
            "status": "SCIENTIFICALLY_VALIDATED",
            "timestamp_utc": stamp(),
            "contract": contract,
            "required_metadata": required,
            "scientific_validation": "PASS",
            "positive_class_semantics": SEMANTICS,
            "probabilities_sha256": file_hash(temporary / "probabilities.npy"),
            "identity_compatibility": {
                "requested_constructor": CONSTRUCTOR.copy(),
                "actual_returned_model_path": required["model_path"],
                "billing_model_version": required["billing_model_version"],
                "execution_mode": required["execution_mode"],
                "synthetic_probe_sha256": PROBE_HASH,
                "reviewed_draft_sha256": DRAFT_HASH,
                "execution_amendment_sha256": amendment_hash,
                "development_identity": "DEVELOPMENT_CONCRETE_IDENTITY_UNAVAILABLE",
                "checkpoint_identical_development_continuity_claimed": False,
            },
        },
    )
    for file in temporary.iterdir():
        file.chmod(0o444)
    if path.exists():
        raise FileExistsError("Checkpoint appeared during publication")
    os.rename(temporary, path)
    path.chmod(0o555)
    mark("PROBABILITY_CHECKPOINT_PUBLISHED")
    # Unexpected optional-adapter exceptions propagate; the validated checkpoint survives.
    mark("OPTIONAL_METADATA_PROCESSING_STARTED")
    optional = optional_metadata(raw)
    mark("OPTIONAL_METADATA_PROCESSED")
    return p[:, 1].copy(), optional


class TransportStopped(BaseException):
    def __init__(self, status=None):
        self.http_status = status if type(status) is int and 100 <= status <= 599 else None
        super().__init__("Transport stopped; no automatic retry")


@contextlib.contextmanager
def one_shot_transport(httpx):
    original = httpx.Client.send
    sequence = 0
    mutex = threading.Lock()

    def send(client, *args, **kwargs):
        nonlocal sequence
        with mutex:
            sequence += 1
            number = sequence
        atomic_json(
            ATTEMPT / "http_events" / f"{number:06d}_STARTED.json",
            {"stage": "HTTP_REQUEST_STARTED", "timestamp_utc": stamp()},
        )
        try:
            response = original(client, *args, **kwargs)
        except Exception:
            raise TransportStopped() from None
        status = response.status_code
        if type(status) is not int or not 100 <= status <= 599:
            raise TransportStopped() from None
        atomic_json(
            ATTEMPT / "http_events" / f"{number:06d}_RETURNED.json",
            {
                "stage": "HTTP_RESPONSE_RECEIVED",
                "http_status": status,
                "http_category": f"HTTP_{status // 100}XX",
                "timestamp_utc": stamp(),
            },
        )
        if status >= 400 and status != 409:
            raise TransportStopped(status) from None
        return response

    httpx.Client.send = send
    try:
        yield
    finally:
        httpx.Client.send = original


@contextlib.contextmanager
def quiet_sdk():
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    saved = [os.dup(1), os.dup(2)]
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                yield
    finally:
        for target, old in zip([1, 2], saved, strict=True):
            os.dup2(old, target)
            os.close(old)
        logging.disable(previous)


def safe_exception(exc):
    allowed = {
        "ScientificFailure",
        "RuntimeError",
        "ValueError",
        "TypeError",
        "PermissionError",
        "FileExistsError",
        "OSError",
        "TransportStopped",
        "KeyboardInterrupt",
        "SystemExit",
    }
    return type(exc).__name__ if type(exc).__name__ in allowed else "OTHER"


def receipt_files(directory):
    return {
        str(path.relative_to(directory)): file_hash(path)
        for path in sorted(Path(directory).rglob("*"))
        if path.is_file()
    }
