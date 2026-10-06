"""One local calibration aggregation; CALIBRATION labels only, seven fixed raw sources."""

import csv
import importlib.abc
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
MODELING = BASE.parent
REPO = MODELING.parents[1]
sys.path.insert(0, str(MODELING / "calibration-full-first-v01"))
import full_recovery_support as full_support  # noqa: E402
from full_recovery_support import (  # noqa: E402,F401
    CONSTRUCTOR,
    LOCK_HASH,
    MAP_AMENDMENT_HASH,
    MAP_CHECKPOINT_HASH,
    MAP_PROBABILITY_HASH,
    MAP_RECEIPT_HASH,
    MAP_SUPERVISOR_HASH,
    PROTOCOL_HASH,
    PYTHON,
    QUERY_ORDER_HASH,
    ScientificFailure,
    array_hash,
    atomic_json,
    digest,
    file_hash,
    quiet_sdk,
    read_json,
    receipt_files,
    row_hash,
    safe_exception,
    seal_json,
    stamp,
    validate_lock,
    verify_seal,
)

from intraop.evaluation.locked_execution import (  # noqa: E402
    CALIBRATED_MODELS as CALIBRATED_MODELS,
)
from intraop.evaluation.locked_execution import (  # noqa: E402
    CALIBRATION_PROTOCOL,
    MODEL_NAMES,
)

FULL_AMENDMENT_HASH = "9e2f973a2ddf190b4485ae6a7e4016f5f11ed3bbaaf33bb074ba630d192dc1e8"
FULL_PROBABILITY_HASH = "7c3edc5533be7f9619e70964ba7ca4ce0d2beb9d27b64e838e118e7f61c42886"
FULL_CHECKPOINT_HASH = "36cabf92bbedf582286a3a88e8ed3a867c0f9d62f6d721be6181f4d7ded1713b"
FULL_RECEIPT_HASH = "15556aa2fbf70f1bc14791558211752049b44d9689f8f700e01a3cc94845fce0"
FULL_SUPERVISOR_HASH = "67fb94f74c8caf94202f40383faaa0b3a733d70e65935acfeedea9c95092bffc"
REGISTRY_HASH = "1fb57f3595eb22ddd1a3ca5b828b9fe79d4904119241628a62ecc979353dffa1"
CALIBRATION = MODELING / "locked_execution/calibration"
ATTEMPT = CALIBRATION / "aggregation/attempt_1"
REGISTRY = BASE / "source_registry.json"
PREPARATION = BASE / "execution_preparation.json"
APPROVAL = MODELING / "locked_execution/approvals/calibration_aggregation.json"
RECEIPT = CALIBRATION / "completion_receipt.json"
ROWS = 2393
POSITIVES = 128
SUBJECTS = 15
FINAL_FILES = {
    "source_registry.json",
    "source_registry.sha256",
    "calibration_parameters.json",
    "calibration_parameters.sha256",
    "calibration_alignment.json",
    "calibration_alignment.sha256",
    "calibration_labels.npy",
    "metrics.json",
    "metrics.sha256",
    "reliability.json",
    "reliability.sha256",
    "reliability.png",
    *[f"predictions/{name}.{suffix}" for name in MODEL_NAMES for suffix in ["csv", "npz"]],
}


class RawRuntimeBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "tabpfn_client",
            "tabpfn",
            "torch",
            "xgboost",
        } or fullname.startswith("intraop.models"):
            raise PermissionError("Raw-model runtimes unavailable to local aggregation")
        return None


def deny_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
        raise PermissionError("Network unavailable to calibration aggregation")


def install_execution_firewall():
    sys.meta_path.insert(0, RawRuntimeBlocker())
    sys.addaudithook(deny_network)


def safe_path(relative):
    path = REPO / relative
    if not path.resolve().is_relative_to(REPO.resolve()) or path.is_symlink():
        raise PermissionError("Source path outside immutable repository")
    return path


def validate_registry(registry, lock, *, verify_sources=True):
    if (
        registry["model_order"] != list(MODEL_NAMES)
        or set(registry["sources"]) != set(MODEL_NAMES)
        or registry["rows"] != ROWS
        or registry["query_source_order_sha256"] != QUERY_ORDER_HASH
        or registry["model_lock_sha256"] != LOCK_HASH
        or registry["calibration_protocol_sha256"] != PROTOCOL_HASH
        or registry["table_hashes"] != lock["table_hashes"]
        or registry["split_manifest_sha256"] != lock["split_manifest_hash"]
    ):
        raise PermissionError("Exact seven-source registry differs")
    for name in MODEL_NAMES:
        record = registry["sources"][name]
        features = lock["models"][name]["features"]
        if (
            record["rows"] != ROWS
            or record["query_source_order_sha256"] != QUERY_ORDER_HASH
            or record["model_lock_sha256"] != LOCK_HASH
            or record["ordered_features"] != features
            or record["feature_count"] != len(features)
            or record["ordered_features_sha256"] != digest(features)
            or record["raw_model_recomputation_permitted"]
            or record["source_kind"]
            != ("DISCRIMINATION_SCORE" if name == "current_map" else "BINARY_PROBABILITY_MATRIX")
            or record["result_shape"] != ([ROWS] if name == "current_map" else [ROWS, 2])
        ):
            raise PermissionError("Source row/feature/model contract differs")
        if name != "current_map" and (
            record["positive_class"] != 1 or record["probability_column"] != 1
        ):
            raise PermissionError("Positive-class contract differs")
        if verify_sources:
            result, manifest = (
                safe_path(record["result_path"]),
                safe_path(record["checkpoint_manifest_path"]),
            )
            if (
                file_hash(result) != record["result_sha256"]
                or file_hash(manifest) != record["checkpoint_manifest_sha256"]
                or result.stat().st_mode & 0o222
                or manifest.stat().st_mode & 0o222
                or result.parent.stat().st_mode & 0o222
            ):
                raise PermissionError("Immutable source hash or permission differs")
            m = read_json(manifest)
            if name == "current_map":
                contract_path = safe_path(record["score_contract_path"])
                if file_hash(contract_path) != record["score_contract_sha256"]:
                    raise PermissionError("Current-MAP contract differs")
                contract = read_json(contract_path)
                expected = m["score_sha256"]
            else:
                contract = m["contract"]
                expected = m["probabilities_sha256"]
                if m["scientific_validation"] != "PASS":
                    raise PermissionError("Unvalidated raw probability source")
            if (
                expected != record["result_sha256"]
                or digest(contract) != record["scientific_contract_sha256"]
            ):
                raise PermissionError("Source contract/result binding differs")
    return registry


def checked_state(*, external=False):
    prep, prep_hash = verify_seal(PREPARATION)
    if (
        prep["scope"] != {"phase": "calibration", "operation": "final_aggregation", "attempt": 1}
        or prep["model_lock_sha256"] != LOCK_HASH
        or prep["calibration_protocol_sha256"] != PROTOCOL_HASH
        or prep["source_registry_sha256"] != REGISTRY_HASH
        or prep["MAP_amendment_sha256"] != MAP_AMENDMENT_HASH
        or prep["full_amendment_sha256"] != FULL_AMENDMENT_HASH
        or prep["raw_models_or_hosted_calls_or_TEST_authorized"]
    ):
        raise PermissionError("Calibration-only preparation scope differs")
    approval, _ = verify_seal(APPROVAL)
    if approval != {
        "status": "APPROVED",
        "scope": prep["scope"],
        "execution_preparation_sha256": prep_hash,
        "source_registry_sha256": REGISTRY_HASH,
        "model_lock_sha256": LOCK_HASH,
        "calibration_protocol_sha256": PROTOCOL_HASH,
        "MAP_amendment_sha256": MAP_AMENDMENT_HASH,
        "full_amendment_sha256": FULL_AMENDMENT_HASH,
        "approval_source_sha256": prep["approval_source_sha256"],
        "one_local_execution_only": True,
        "automatic_retry": False,
        "raw_models_authorized": False,
        "hosted_calls_authorized": False,
        "TEST_authorized": False,
    }:
        raise PermissionError("Exact aggregation authorization differs")
    for relative, expected in prep["bound_files_sha256"].items():
        if file_hash(safe_path(relative)) != expected:
            raise PermissionError("Hash-bound execution source differs")
    for relative, record in prep["protected_inventory"].items():
        path = safe_path(relative)
        if (
            file_hash(path) != record["sha256"]
            or path.stat().st_size != record["bytes"]
            or oct(path.stat().st_mode & 0o777) != record["mode"]
        ):
            raise PermissionError("Preserved worker/evidence differs")
    execution = MODELING / "locked_execution"
    if any(p.exists() for p in [execution / "test", execution / "approvals/test.json"]):
        raise PermissionError("TEST unavailable before calibration loading")
    if (
        RECEIPT.exists()
        or (CALIBRATION / "aggregation").exists()
        and any(p.name != "attempt_1" for p in (CALIBRATION / "aggregation").iterdir())
    ):
        raise PermissionError("Completed aggregation or extra attempt retained; no retry")
    for p in [CALIBRATION, ATTEMPT.parent, ATTEMPT]:
        if p.is_symlink():
            raise PermissionError("Execution paths must not be symlinks")
    protected = set(prep["protected_inventory"])
    for p in CALIBRATION.rglob("*"):
        if p.is_file() and not p.is_relative_to(ATTEMPT):
            relative = str(p.relative_to(CALIBRATION))
            if str(p.relative_to(REPO)) not in protected and relative not in FINAL_FILES:
                raise PermissionError("Unreviewed calibration state")
    verify_seal(
        MODELING / "calibration-map-recovery-v01/execution_amendment.json", MAP_AMENDMENT_HASH
    )
    verify_seal(
        MODELING / "calibration-full-first-v01/execution_amendment.json", FULL_AMENDMENT_HASH
    )
    full_root = CALIBRATION / "workers/tabpfn_full"
    verify_seal(full_root / "completion_receipt.json", FULL_RECEIPT_HASH)
    verify_seal(full_root / "supervisor_receipt.json", FULL_SUPERVISOR_HASH)
    full_support.verify_map_completion()
    lock, actual = validate_lock(MODELING, expected=LOCK_HASH, check_environment=True)
    protocol, ph = verify_seal(MODELING / "calibration_protocol.json", PROTOCOL_HASH)
    if actual != LOCK_HASH or ph != PROTOCOL_HASH or protocol != CALIBRATION_PROTOCOL:
        raise PermissionError("Prespecified calibration procedure differs")
    registry, _ = verify_seal(REGISTRY, REGISTRY_HASH)
    validate_registry(registry, lock)
    if external and (
        sys.executable != PYTHON
        or os.environ.get("INTRAOP_AGGREGATION_EXECUTION") != "normal_terminal"
    ):
        raise PermissionError("Exact normal-terminal interpreter boundary required")
    return prep, prep_hash, lock, registry


def reserve():
    prep, ph, lock, registry = checked_state(external=True)
    if ATTEMPT.exists() or any((CALIBRATION / name).exists() for name in FINAL_FILES):
        raise FileExistsError("Existing aggregate state retained; no duplicate or retry")
    ATTEMPT.mkdir(parents=True, exist_ok=False)
    atomic_json(
        ATTEMPT / "reservation.json",
        {
            "status": "RESERVED",
            "attempt": 1,
            "scope": prep["scope"],
            "execution_preparation_sha256": ph,
            "source_registry_sha256": REGISTRY_HASH,
            "timestamp_utc": stamp(),
            "automatic_retry": False,
        },
    )
    return ph


class CalibrationLabels:
    """Only CALIBRATION labels may be routed; no predictor table is parsed."""

    def __init__(self):
        _, self.preparation_hash, self.lock, self.registry = checked_state(external=True)
        self.access_log = []

    def authorize(self):
        _, ph, lock, registry = checked_state(external=True)
        reservation = read_json(ATTEMPT / "reservation.json")
        if (
            ph != self.preparation_hash
            or reservation["status"] != "RESERVED"
            or reservation["execution_preparation_sha256"] != ph
            or (ATTEMPT / "failure.json").exists()
            or (ATTEMPT / "supervisor_stop.json").exists()
            or (ATTEMPT / "worker_completion.json").exists()
        ):
            raise PermissionError("Prior or changed execution cannot continue")
        return lock

    def load(self, partition):
        if partition != "calibration":
            raise PermissionError("Only CALIBRATION labels are available")
        self.authorize()
        cohort = REPO / self.lock["cohort_checkpoint"]
        split = read_json(cohort / "realized_split_manifest.json")
        y, index = read_calibration_labels(cohort, split)
        self.access_log.append("calibration")
        return y, index


def read_calibration_labels(cohort, split):
    permitted = [
        subject
        for subject, partition in split["subject_to_partition"].items()
        if partition == "calibration"
    ]
    if len(permitted) != SUBJECTS:
        raise ValueError("Frozen CALIBRATION roster differs")
    selected = []
    with (Path(cohort) / "metadata.csv").open() as stream:
        header = next(csv.reader([next(stream)]))
        position = header.index("subject_id")
        roster = "|".join(re.escape(subject) for subject in permitted)
        route = re.compile(r"^(?:[^,]*,){" + str(position) + r"}(?:" + roster + r")(?=,)")
        for index, line in enumerate(stream):
            if route.match(line):
                selected.append(index)
        metadata_rows = index + 1
    chosen = set(selected)
    labels = []
    with (Path(cohort) / "labels.csv").open() as stream:
        if next(csv.reader([next(stream)])) != ["label"]:
            raise ValueError("Frozen label schema differs")
        for index, line in enumerate(stream):
            if index in chosen:
                value = next(csv.reader([line]))
                if len(value) != 1 or value[0] not in ("0", "1"):
                    raise ValueError("Frozen binary label differs")
                labels.append(int(value[0]))
        if index + 1 != metadata_rows:
            raise ValueError("Label/metadata row counts differ")
    y = np.asarray(labels, dtype=int)
    order = np.asarray(selected, dtype=np.int64)
    if (
        len(y) != ROWS
        or row_hash(order) != QUERY_ORDER_HASH
        or int(y.sum()) != POSITIVES
        or np.unique(y).tolist() != [0, 1]
    ):
        raise ValueError("Frozen CALIBRATION labels/order differ")
    return y, order


def raw_outputs(registry):
    values = {}
    for name in MODEL_NAMES:
        record = registry["sources"][name]
        path = safe_path(record["result_path"])
        if file_hash(path) != record["result_sha256"]:
            raise PermissionError("Source changed before read")
        a = np.load(path, allow_pickle=False)
        if a.shape != tuple(record["result_shape"]) or not np.isfinite(a).all():
            raise ValueError("Raw result numeric shape differs")
        if name == "current_map":
            values[name] = a.copy()
        else:
            if (
                (a < 0).any()
                or (a > 1).any()
                or not np.allclose(a.sum(axis=1), 1, atol=1e-6, rtol=0)
            ):
                raise ValueError("Raw binary probabilities invalid")
            values[name] = a[:, 1].copy()
    return values


def atomic_binary(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".publication-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(tmp, path)
        path.chmod(0o444)
    finally:
        Path(tmp).unlink(missing_ok=True)


def copy_immutable(source, destination):
    atomic_binary(destination, lambda stream: stream.write(Path(source).read_bytes()))
