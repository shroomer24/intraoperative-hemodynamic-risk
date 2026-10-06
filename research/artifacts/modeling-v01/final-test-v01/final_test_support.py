"""One reserved external TEST evaluation, immutable scientific and calibration state."""

import contextlib
import importlib.abc
import os
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
MODELING = BASE.parent
REPO = MODELING.parents[1]
sys.path.insert(0, str(MODELING / "calibration-aggregation-v01"))
import aggregation_support as a  # noqa: E402

from intraop.evaluation.locked_execution import (  # noqa: E402
    TEST_PROTOCOL,
    verify_calibration_complete,
)

ROOT = MODELING / "locked_execution/test"
RESERVATION = ROOT / "execution_manifest.json"
RECEIPT = ROOT / "completion_receipt.json"
APPROVAL = MODELING / "locked_execution/approvals/test.json"
TEMPLATE = BASE / "test_approval_TEMPLATE.json"
PREPARATION = BASE / "execution_preparation.json"
REGISTRY = BASE / "source_registry.json"
REGISTRY_HASH = "d0198454bc74ff9382be5ab17bea5aafa0cd46179f8514155de78f111ee79fcd"
CALIBRATION_RECEIPT_HASH = "9d72c180324fbc847958d027f20abb20689a7b74e47d01b334c9abfb1f7741c9"
MAPPINGS_HASH = "6cff372cb831a060cf3457c7c74d3b05d557a2be64a26cc6581016d7e15f5ba2"
TEST_PROTOCOL_HASH = "521dbcb48ee495504e565621e67d489e479af6bf3469306502352b405b60a9dc"
MODELS = a.MODEL_NAMES
LEARNED = a.CALIBRATED_MODELS
HOSTED = ("tabpfn_map", "tabpfn_full")
MAP_HASH = "32ca198f264e92caa36d101fc83d1f9efc72e3bee3ca4b400be0c12d4716ab26"
FULL_HASH = "3de7743564650c9e138ac30651844bf7089a66713398606b4dd464e2bee80f67"
DECLARED_MAPPINGS = {
    "logistic_map": (0.7297439205179267, -0.5686494994216078),
    "logistic_full": (0.650705561096738, -0.8078187767952033),
    "xgboost": (0.9515966466147724, -0.020615578392713343),
    "tabpfn_map": (0.7056409052253757, -0.5661993467267635),
    "tabpfn_full": (1.0594559064699107, -0.029297148344735954),
}


def verify_mappings(mappings):
    if set(mappings) != set(LEARNED):
        raise PermissionError("Exactly the five sealed calibration mappings required")
    for name, pair in DECLARED_MAPPINGS.items():
        p = mappings[name]
        if (
            (p["coefficient"], p["intercept"]) != pair
            or p["epsilon"] != 1e-6
            or p["method"] != "Platt_logit_logistic"
            or p["fit_partition"] != "calibration"
            or p["configuration_changed"]
            or p["fitting_parameters"] != a.CALIBRATION_PROTOCOL["parameters"]
        ):
            raise PermissionError("Sealed Platt mapping differs")
    return mappings


def expected_approval(prep, preparation_hash):
    return {
        "status": "APPROVED",
        "phase": "test",
        "model_lock_sha256": a.LOCK_HASH,
        "protocol_sha256": TEST_PROTOCOL_HASH,
        "one_execution_only": True,
        "execution_preparation_sha256": preparation_hash,
        "source_registry_sha256": REGISTRY_HASH,
        "calibration_completion_receipt_sha256": CALIBRATION_RECEIPT_HASH,
        "calibration_parameters_sha256": MAPPINGS_HASH,
        "approval_source_sha256": prep["approval_source_sha256"],
        "automatic_retry": False,
        "calibration_refit_authorized": False,
        "configuration_changes_authorized": False,
    }


def checked_state(*, external=False, live=False):
    prep, ph = a.verify_seal(PREPARATION)
    if (
        prep["scope"] != {"phase": "test", "attempt": 1}
        or prep["model_lock_sha256"] != a.LOCK_HASH
        or prep["calibration_completion_receipt_sha256"] != CALIBRATION_RECEIPT_HASH
        or prep["calibration_parameters_sha256"] != MAPPINGS_HASH
        or prep["source_registry_sha256"] != REGISTRY_HASH
        or prep["test_protocol_sha256"] != TEST_PROTOCOL_HASH
        or prep["calibration_refit_authorized"]
        or prep["scientific_configuration_changes_authorized"]
    ):
        raise PermissionError("Approved one-time TEST preparation differs")
    expected = expected_approval(prep, ph)
    if a.verify_seal(TEMPLATE)[0] != expected:
        raise PermissionError("Exact architectural approval template required")
    if live and a.verify_seal(APPROVAL)[0] != expected:
        raise PermissionError("Installed TEST architectural approval required")
    for rel, h in prep["bound_files_sha256"].items():
        if a.file_hash(a.safe_path(rel)) != h:
            raise PermissionError("Exact execution source hash differs")
    for rel, r in prep["protected_inventory"].items():
        p = a.safe_path(rel)
        if (
            a.file_hash(p) != r["sha256"]
            or p.stat().st_size != r["bytes"]
            or oct(p.stat().st_mode & 0o777) != r["mode"]
        ):
            raise PermissionError("Preserved scientific/calibration evidence differs")
    lock, lh = a.validate_lock(MODELING, expected=a.LOCK_HASH, check_environment=True)
    a.verify_seal(a.RECEIPT, CALIBRATION_RECEIPT_HASH)
    verify_calibration_complete(MODELING, lh)
    if a.verify_seal(MODELING / "test_protocol.json", TEST_PROTOCOL_HASH)[0] != TEST_PROTOCOL:
        raise PermissionError("Frozen TEST protocol differs")
    a.verify_seal(MODELING / "calibration_protocol.json", a.PROTOCOL_HASH)
    mappings = verify_mappings(
        a.verify_seal(a.CALIBRATION / "calibration_parameters.json", MAPPINGS_HASH)[0]
    )
    registry, _ = a.verify_seal(REGISTRY, REGISTRY_HASH)
    if (
        registry["models"] != list(MODELS)
        or registry["training_context"] != lock["training_context"]
    ):
        raise PermissionError("Frozen model/context registry differs")
    for name in MODELS:
        record, definition = registry["sources"][name], lock["models"][name]
        if (
            record["locked_definition_sha256"] != a.digest(definition)
            or record["features"] != definition["features"]
            or record["feature_contract_sha256"] != a.digest(definition["features"])
            or record["prespecified_Platt_mapping"] != mappings.get(name)
        ):
            raise PermissionError("Locked model/features/mapping differs")
        if record["fitted_model_path"]:
            p = a.safe_path(record["fitted_model_path"])
            if a.file_hash(p) != record["fitted_model_sha256"] or p.stat().st_mode & 0o222:
                raise PermissionError("Preserved fitted model differs")
    if RECEIPT.exists() or (ROOT / "failure.json").exists():
        raise PermissionError("Prior TEST completion/failure retained; no retry")
    if ROOT.is_symlink() or ROOT.parent.is_symlink():
        raise PermissionError("TEST execution path must not be a symlink")
    if external and (
        sys.executable != a.PYTHON
        or os.environ.get("INTRAOP_FINAL_TEST_EXECUTION") != "normal_terminal"
        or not os.environ.get("TABPFN_TOKEN")
    ):
        raise PermissionError("Exact normal-terminal interpreter/credential boundary required")
    return prep, ph, lock, registry, mappings


def reserve():
    prep, ph, lock, registry, mappings = checked_state(external=True)
    if ROOT.exists() or ROOT.is_symlink():
        raise FileExistsError("Any existing TEST execution state blocks rerun")
    # Approval is installed only by the human-invoked external launcher.
    if APPROVAL.exists():
        if a.verify_seal(APPROVAL)[0] != expected_approval(prep, ph):
            raise PermissionError("Existing TEST approval differs")
    else:
        a.copy_immutable(TEMPLATE, APPROVAL)
        a.copy_immutable(TEMPLATE.with_suffix(".sha256"), APPROVAL.with_suffix(".sha256"))
    checked_state(external=True, live=True)
    ROOT.mkdir(exist_ok=False)
    a.seal_json(
        RESERVATION,
        {
            "status": "RESERVED",
            "phase": "test",
            "attempt": 1,
            "timestamp_utc": a.stamp(),
            "execution_preparation_sha256": ph,
            "source_registry_sha256": REGISTRY_HASH,
            "model_lock_sha256": a.LOCK_HASH,
            "calibration_completion_receipt_sha256": CALIBRATION_RECEIPT_HASH,
            "calibration_parameters_sha256": MAPPINGS_HASH,
            "automatic_retry": False,
        },
    )
    return prep, ph, lock, registry, mappings


class TestPartitions:
    """TRAINING-only context and TEST-only query, after live approval/reservation."""

    def __init__(self):
        _, self.preparation_hash, self.lock, self.registry, self.mappings = checked_state(
            external=True, live=True
        )
        self.cohort = REPO / self.lock["cohort_checkpoint"]
        self.access_log = []

    def authorize(self):
        _, ph, lock, registry, mappings = checked_state(external=True, live=True)
        reservation, _ = a.verify_seal(RESERVATION)
        if (
            ph != self.preparation_hash
            or reservation["status"] != "RESERVED"
            or reservation["phase"] != "test"
            or reservation["attempt"] != 1
            or reservation["execution_preparation_sha256"] != ph
            or reservation["model_lock_sha256"] != a.LOCK_HASH
            or reservation["source_registry_sha256"] != REGISTRY_HASH
            or reservation["automatic_retry"]
        ):
            raise PermissionError("Exact one-time TEST reservation required before I/O")
        return lock

    def load(self, partition):
        if partition not in ("training", "test"):
            raise PermissionError("TUNING/CALIBRATION cannot enter TEST evaluation")
        self.authorize()
        split = a.read_json(self.cohort / "realized_split_manifest.json")
        data = a.full_support._load_partition(
            self.cohort, split, self.lock["feature_sets"]["full"], partition
        )
        roster = {s for s, p in split["subject_to_partition"].items() if p == partition}
        if set(data.groups) != roster or (partition == "test" and len(roster) != 22):
            raise PermissionError("Exact frozen subject roster required")
        self.access_log.append(partition)
        return data


def input_contract(loader, name, training, query):
    if name not in MODELS:
        raise PermissionError("Only seven locked TEST models are available")
    definition = loader.lock["models"][name]
    record = loader.registry["sources"][name]
    context = loader.lock["training_context"]
    features = definition["features"]
    X = a.full_support.predictor_array(training, features)
    q = a.full_support.predictor_array(query, features)
    if (
        len(training.X) != context["rows"]
        or training.groups.nunique() != context["subjects"]
        or int(training.y.sum()) != context["positive_labels"]
        or np.unique(training.y).tolist() != [0, 1]
        or set(training.groups) & set(query.groups)
        or a.array_hash(X) != record["training_X_sha256"]
        or a.array_hash(training.y) != record["training_y_sha256"]
        or a.digest(definition) != record["locked_definition_sha256"]
    ):
        raise a.ScientificFailure("input_contract", None, "TRAINING_CONTEXT_MISMATCH")
    if name in HOSTED and (
        definition["selected_configuration"]["hyperparameters"] != a.CONSTRUCTOR
        or len(features) != (18 if name == "tabpfn_map" else 74)
        or a.digest(features) != (MAP_HASH if name == "tabpfn_map" else FULL_HASH)
    ):
        raise a.ScientificFailure("features", None, "FROZEN_CONTRACT_MISMATCH")
    contract = {
        "phase": "test",
        "candidate": name,
        "kind": definition["kind"],
        "constructor": definition["selected_configuration"]["hyperparameters"],
        "features": features,
        "feature_count": len(features),
        "feature_contract_sha256": a.digest(features),
        "training_X_sha256": a.array_hash(X),
        "training_y_sha256": a.array_hash(training.y),
        "training_context_sha256": a.digest(context),
        "model_definition_sha256": a.digest(definition),
        "query_X_sha256": a.array_hash(q),
        "query_rows": len(query.X),
        "query_source_order_sha256": a.row_hash(query.X.index),
        "query_order_verified": bool(
            query.X.index.is_unique
            and query.X.index.is_monotonic_increasing
            and query.X.index.equals(query.y.index)
            and query.X.index.equals(query.metadata.index)
        ),
        "metadata_excluded_from_X": not bool(set(features) & set(query.metadata)),
        "model_lock_sha256": a.LOCK_HASH,
        "dataset_hashes": loader.lock["table_hashes"],
        "split_manifest_sha256": loader.lock["split_manifest_hash"],
        "software_versions": loader.lock["package_versions"],
        "positive_class": 1,
        "probability_column": 1,
        "positive_class_semantics": a.full_support.SEMANTICS,
        "source_registry_sha256": REGISTRY_HASH,
    }
    return X, q, contract


def checkpoint(path, probabilities, classes, contract, observed, raw, *, mark):
    mark("PROBABILITY_VALIDATION_STARTED")
    cls = np.asarray(classes)
    if cls.dtype.kind not in "iu" or cls.tolist() != [0, 1]:
        raise a.ScientificFailure("classes", None, "ORDERED_BINARY_CLASSES_REQUIRED")
    if contract["phase"] != "test" or contract["source_registry_sha256"] != REGISTRY_HASH:
        raise PermissionError("TEST-only checkpoint required")
    p = a.full_support.validate_probabilities(probabilities, classes, contract, observed)
    required = (
        a.full_support.compatible_metadata(raw, rows=len(p), features=contract["feature_count"])
        if contract["kind"] == "tabpfn"
        else {}
    )
    mark("PROBABILITY_VALIDATION_PASSED")
    path = Path(path)
    if path.exists():
        raise FileExistsError("Immutable TEST checkpoint already exists")
    a.atomic_json(
        path.parent / (path.name + ".reservation.json"),
        {
            "status": "RESERVED",
            "scientific_contract_sha256": a.digest(contract),
        },
    )
    temp = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=path.parent))
    a.atomic_binary(
        temp / "probabilities.npy", lambda stream: np.save(stream, p, allow_pickle=False)
    )
    a.seal_json(
        temp / "manifest.json",
        {
            "status": "SCIENTIFICALLY_VALIDATED",
            "scientific_validation": "PASS",
            "contract": contract,
            "required_metadata": required,
            "probabilities_sha256": a.file_hash(temp / "probabilities.npy"),
            "classes": [0, 1],
            "positive_class": 1,
            "probability_column": 1,
            "timestamp_utc": a.stamp(),
        },
    )
    if path.exists():
        raise FileExistsError("Checkpoint appeared during publication")
    os.rename(temp, path)
    path.chmod(0o555)
    mark("PROBABILITY_CHECKPOINT_PUBLISHED")
    mark("OPTIONAL_METADATA_PROCESSING_STARTED")
    optional = a.full_support.optional_metadata(raw)
    mark("OPTIONAL_METADATA_PROCESSED")
    return p[:, 1].copy(), optional


class RuntimeBlocker(importlib.abc.MetaPathFinder):
    def __init__(self, name):
        self.forbidden = (
            {"torch", "tabpfn", "tabpfn_client"}
            if name == "xgboost"
            else {"xgboost"}
            if name in HOSTED
            else {"torch", "tabpfn", "tabpfn_client", "xgboost"}
        )

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.forbidden:
            raise PermissionError("Incompatible runtime unavailable in isolated TEST worker")
        return None


@contextlib.contextmanager
def one_shot_transport(httpx, output):
    original, sequence, mutex = httpx.Client.send, 0, threading.Lock()

    def send(client, *args, **kwargs):
        nonlocal sequence
        with mutex:
            sequence += 1
            number = sequence
        a.atomic_json(
            output / "http_events" / f"{number:06d}_STARTED.json",
            {
                "stage": "HTTP_REQUEST_STARTED",
                "timestamp_utc": a.stamp(),
            },
        )
        try:
            response = original(client, *args, **kwargs)
        except Exception:
            raise a.full_support.TransportStopped() from None
        status = response.status_code
        if type(status) is not int or not 100 <= status <= 599:
            raise a.full_support.TransportStopped() from None
        a.atomic_json(
            output / "http_events" / f"{number:06d}_RETURNED.json",
            {
                "stage": "HTTP_RESPONSE_RECEIVED",
                "http_status": status,
                "http_category": f"HTTP_{status // 100}XX",
                "timestamp_utc": a.stamp(),
            },
        )
        if status >= 400 and status != 409:
            raise a.full_support.TransportStopped(status) from None
        return response

    httpx.Client.send = send
    try:
        yield
    finally:
        httpx.Client.send = original
