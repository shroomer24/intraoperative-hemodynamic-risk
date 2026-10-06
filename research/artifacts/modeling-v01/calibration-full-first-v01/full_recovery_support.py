"""First full calibration worker only; immutable MAP and scientific state preserved."""

import contextlib
import os
import shlex
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
MODELING = BASE.parent
REPO = MODELING.parents[1]
MAP_BASE = MODELING / "calibration-map-recovery-v01"
sys.path.insert(0, str(MAP_BASE))
import map_recovery_support as map_support  # noqa: E402

if Path(map_support.__file__).resolve() != MAP_BASE / "map_recovery_support.py":
    raise PermissionError("Approved identity-validator import differs")

from map_recovery_support import (  # noqa: E402,F401
    CANONICAL,
    CONSTRUCTOR,
    DRAFT_HASH,
    LOCK_HASH,
    PROBE_HASH,
    PROTOCOL_HASH,
    QUERY_ORDER_HASH,
    SEMANTICS,
    SUPERVISOR_FAILURE_HASH,
    WORKER_FAILURE_HASH,
    ScientificFailure,
    TransportStopped,
    _load_partition,
    array_hash,
    atomic_json,
    compatible_metadata,
    digest,
    file_hash,
    optional_metadata,
    predictor_array,
    quiet_sdk,
    read_json,
    receipt_files,
    row_hash,
    safe_exception,
    seal_json,
    stamp,
    validate_lock,
    validate_probabilities,
    verify_seal,
)

PYTHON = "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3"
FEATURE_HASH = "3de7743564650c9e138ac30651844bf7089a66713398606b4dd464e2bee80f67"
CONTEXT_HASH = "e34366859f3201970f06cfa7bc75d45f53049d8e0152364b6fb7da5412d04ece"
MODEL_HASH = "048bb7d142f28b4ace949b20b338f99774763d6e051c90e7ce705c391230f42e"
MAP_AMENDMENT_HASH = "fea7ee46226c6c73817a7144cd33d23566ddae781b3b025e3664f845ec4f941d"
MAP_PROBABILITY_HASH = "543c194bbe7252f3d7299b52ebe4603abdbf055186676994809d58e86632beec"
MAP_CHECKPOINT_HASH = "6fa111702b2571843d3bb7a9872019edd3280b36721a261318ca90240c009599"
MAP_RECEIPT_HASH = "b968aec9c6c095b577151217be241ccc5382ec6589dda47c99cef40ddf969cbf"
MAP_SUPERVISOR_HASH = "6af668872d28d9cf4b85b74afc93e99e486adae14dcf81f53f013e1cbf5d55af"
CALIBRATION = MODELING / "locked_execution/calibration"
MAP_ATTEMPT = CALIBRATION / "attempts/tabpfn_map/attempt_2"
ATTEMPT = CALIBRATION / "workers/tabpfn_full"
AMENDMENT = BASE / "execution_amendment.json"
APPROVAL = MODELING / "locked_execution/approvals/calibration_full_first.json"
PRESERVED = ("prevalence", "current_map", "logistic_map", "logistic_full", "xgboost", "tabpfn_map")


def resolve_attempt(model="tabpfn_full", number=1):
    if model != "tabpfn_full" or type(number) is not int or number != 1:
        raise PermissionError("Only the first full calibration execution is approved")
    return ATTEMPT


def verify_map_completion():
    receipt, _ = verify_seal(MAP_ATTEMPT / "completion_receipt.json", MAP_RECEIPT_HASH)
    supervisor, _ = verify_seal(MAP_ATTEMPT / "supervisor_receipt.json", MAP_SUPERVISOR_HASH)
    manifest, _ = verify_seal(
        MAP_ATTEMPT / "probability_checkpoint/manifest.json", MAP_CHECKPOINT_HASH
    )
    process = read_json(MAP_ATTEMPT / "process_exit.json")
    if (
        receipt["status"] != "COMPLETE"
        or receipt["model"] != "tabpfn_map"
        or receipt["attempt"] != 2
        or receipt["execution_amendment_sha256"] != MAP_AMENDMENT_HASH
        or receipt["partition_access"] != ["training", "calibration"]
        or supervisor["status"] != "COMPLETE_MAP_ONLY_STOP_FOR_REVIEW"
        or supervisor["worker_completion_receipt_sha256"] != MAP_RECEIPT_HASH
        or supervisor["execution_amendment_sha256"] != MAP_AMENDMENT_HASH
        or process["exit_code"] != 0
        or process["signal"] is not None
        or manifest["scientific_validation"] != "PASS"
        or manifest["probabilities_sha256"] != MAP_PROBABILITY_HASH
        or file_hash(MAP_ATTEMPT / "probability_checkpoint/probabilities.npy")
        != MAP_PROBABILITY_HASH
        or (MAP_ATTEMPT / "failure.json").exists()
        or (MAP_ATTEMPT / "supervisor_stop.json").exists()
    ):
        raise PermissionError("Validated MAP completion is required unchanged")
    return receipt


def checked_state(*, external=False):
    amendment, amendment_hash = verify_seal(AMENDMENT)
    if (
        amendment["scope"] != {"phase": "calibration", "model": "tabpfn_full", "attempt": 1}
        or amendment["model_lock_sha256"] != LOCK_HASH
        or amendment["calibration_protocol_sha256"] != PROTOCOL_HASH
        or amendment["synthetic_probe_sha256"] != PROBE_HASH
        or amendment["reviewed_draft_sha256"] != DRAFT_HASH
        or amendment["map_compatibility_amendment_sha256"] != MAP_AMENDMENT_HASH
        or amendment["map_probability_sha256"] != MAP_PROBABILITY_HASH
        or amendment["map_checkpoint_manifest_sha256"] != MAP_CHECKPOINT_HASH
        or amendment["development_identity"] != "DEVELOPMENT_CONCRETE_IDENTITY_UNAVAILABLE"
        or amendment["checkpoint_identical_development_continuity_claimed"]
        or amendment["Platt_or_TEST_execution_authorized"]
    ):
        raise PermissionError("Approved first-full scope differs")
    approval, _ = verify_seal(APPROVAL)
    if approval != {
        "status": "APPROVED",
        "scope": {"phase": "calibration", "model": "tabpfn_full", "attempt": 1},
        "execution_amendment_sha256": amendment_hash,
        "model_lock_sha256": LOCK_HASH,
        "calibration_protocol_sha256": PROTOCOL_HASH,
        "synthetic_probe_sha256": PROBE_HASH,
        "reviewed_draft_sha256": DRAFT_HASH,
        "map_compatibility_amendment_sha256": MAP_AMENDMENT_HASH,
        "map_probability_sha256": MAP_PROBABILITY_HASH,
        "map_checkpoint_manifest_sha256": MAP_CHECKPOINT_HASH,
        "approval_source_sha256": amendment["approval_source_sha256"],
        "maximum_fit_calls": 1,
        "maximum_predict_proba_calls": 1,
        "automatic_retry": False,
        "second_full_attempt_authorized": False,
        "Platt_authorized": False,
        "TEST_authorized": False,
    }:
        raise PermissionError("Specific architectural authorization differs")
    verify_seal(MAP_BASE / "execution_amendment.json", MAP_AMENDMENT_HASH)
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
    protected = set(amendment["protected_inventory"])
    for path in CALIBRATION.rglob("*"):
        if path.is_file() and not path.is_relative_to(ATTEMPT):
            if str(path.relative_to(REPO)) not in protected:
                raise PermissionError("Unreviewed calibration execution state")
    for relative, expected in {
        "workers/tabpfn_map/failure.json": WORKER_FAILURE_HASH,
        "failure.json": SUPERVISOR_FAILURE_HASH,
    }.items():
        if file_hash(CALIBRATION / relative) != expected:
            raise PermissionError("Original reviewed failure evidence differs")
    verify_map_completion()
    execution = MODELING / "locked_execution"
    if any(
        path.exists()
        for path in [
            execution / "test",
            execution / "approvals/test.json",
            CALIBRATION / "completion_receipt.json",
            CALIBRATION / "calibrators",
            CALIBRATION / "attempts/tabpfn_full",
        ]
    ):
        raise PermissionError("TEST, Platt, phase completion or a second attempt unavailable")
    for path in [CALIBRATION, ATTEMPT.parent, ATTEMPT]:
        if path.is_symlink():
            raise PermissionError("Execution paths must not be symlinks")
    lock, actual = validate_lock(MODELING, expected=LOCK_HASH, check_environment=True)
    verify_seal(MODELING / "calibration_protocol.json", PROTOCOL_HASH)
    definition = lock["models"]["tabpfn_full"]
    if (
        actual != LOCK_HASH
        or lock["tabpfn_representation"] != "A_frozen74_retained_control"
        or digest(definition) != MODEL_HASH
        or digest(lock["training_context"]) != CONTEXT_HASH
        or definition["selected_configuration"]["hyperparameters"] != CONSTRUCTOR
        or definition["features"] != lock["feature_sets"]["full"]
        or len(definition["features"]) != 74
        or digest(definition["features"]) != FEATURE_HASH
    ):
        raise PermissionError("Only the locked 74-feature baseline A is available")
    if external and (
        sys.executable != PYTHON
        or os.environ.get("INTRAOP_FULL_FIRST_EXECUTION") != "normal_terminal"
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
        "full_worker.py",
        "run_full_first.py",
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
        raise FileExistsError("Prior first-full execution is retained; no duplicate or retry")
    if process_check.get("status") != "NO_RELEVANT_WORKER_ACTIVE":
        raise PermissionError("Live worker inactivity is required")
    path.mkdir(parents=True, exist_ok=False)
    reservation = {
        "status": "RESERVED",
        "phase": "calibration",
        "model": "tabpfn_full",
        "attempt": 1,
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
            or reservation["model"] != "tabpfn_full"
            or reservation["attempt"] != 1
            or (ATTEMPT / "failure.json").exists()
            or (ATTEMPT / "completion_receipt.json").exists()
        ):
            raise PermissionError("New failed/complete/indeterminate attempt cannot repeat")
        return lock

    def load(self, partition):
        if partition not in ("training", "calibration"):
            raise PermissionError("Partition unavailable to first-full recovery")
        self.authorize()  # before routing any table lines
        split = read_json(self.cohort / "realized_split_manifest.json")
        data = _load_partition(self.cohort, split, self.lock["feature_sets"]["full"], partition)
        self.access_log.append(partition)
        return data


def input_contract(loader, training, query):
    context = loader.lock["training_context"]
    features = loader.lock["models"]["tabpfn_full"]["features"]
    if (
        digest(context) != CONTEXT_HASH
        or digest(loader.lock["models"]["tabpfn_full"]) != MODEL_HASH
        or len(training.X) != context["rows"]
        or training.groups.nunique() != context["subjects"]
        or int(training.y.sum()) != context["positive_labels"]
        or np.unique(training.y).tolist() != [0, 1]
        or set(training.groups) & set(query.groups)
        or len(query.X) != 2393
        or row_hash(query.X.index) != QUERY_ORDER_HASH
        or len(features) != 74
        or digest(features) != FEATURE_HASH
    ):
        raise ScientificFailure("input_contract", None, "FROZEN_CONTEXT_MISMATCH")
    X, query_X = predictor_array(training, features), predictor_array(query, features)
    contract = {
        "candidate": "tabpfn_full",
        "kind": "tabpfn",
        "constructor": CONSTRUCTOR.copy(),
        "features": features,
        "feature_count": 74,
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
        "training_context_sha256": digest(loader.lock["training_context"]),
        "model_definition_sha256": digest(loader.lock["models"]["tabpfn_full"]),
        "dataset_hashes": loader.lock["table_hashes"],
        "split_manifest_sha256": loader.lock["split_manifest_hash"],
        "software_versions": loader.lock["package_versions"],
        "positive_class": 1,
        "probability_column": 1,
        "positive_class_semantics": SEMANTICS,
    }
    return X, query_X, contract


def persist_approved_result(
    path, probabilities, classes, contract, observed, raw, *, amendment_hash, mark
):
    mark("PROBABILITY_VALIDATION_STARTED")
    a = np.asarray(classes)
    if a.dtype.kind not in "iu" or a.shape != (2,) or a.tolist() != [0, 1]:
        raise ScientificFailure("classes", None, "ORDERED_BINARY_CLASSES_REQUIRED")
    if (
        contract.get("candidate") != "tabpfn_full"
        or contract.get("kind") != "tabpfn"
        or contract.get("model_lock_sha256") != LOCK_HASH
        or contract.get("feature_count") != 74
        or digest(contract.get("features")) != FEATURE_HASH
        or contract.get("feature_contract_sha256") != FEATURE_HASH
        or contract.get("training_context_sha256") != CONTEXT_HASH
        or contract.get("model_definition_sha256") != MODEL_HASH
    ):
        raise ScientificFailure("input_contract", None, "FROZEN_FULL_FIRST_SCOPE_MISMATCH")
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
                "map_compatibility_amendment_sha256": MAP_AMENDMENT_HASH,
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
