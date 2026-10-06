"""Scientific validation precedes immutable persistence and optional provenance.

No SDK is imported. Only closed field names/types/reason codes enter diagnostics.
Returned arrays use the SDK positional query contract; no extra hosted queries.
"""

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

CONSTRUCTOR = {"model_path": "v3.5_default", "random_state": 42}
SEMANTICS = "1 = new sustained invasive MAP <65 mmHg episode lasting >=60s starting in (t,t+300s]"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def array_hash(value):
    a = np.array(value, dtype="<f8", order="C", copy=True)
    a[np.isnan(a)] = np.nan
    return hashlib.sha256(a.tobytes()).hexdigest()


def row_hash(index):
    return hashlib.sha256(np.asarray(index, dtype="<i8").tobytes()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".atomic-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # no replacement, including concurrent writers
        path.chmod(0o444)
    finally:
        Path(temporary).unlink(missing_ok=True)


def safe_type(value):
    for kind, label in [
        (dict, "mapping"),
        (list, "list"),
        (tuple, "tuple"),
        (str, "string"),
        (bool, "boolean"),
        (int, "integer"),
        (float, "float"),
        (np.ndarray, "array"),
    ]:
        if isinstance(value, kind):
            return label
    return "null" if value is None else "OTHER"


class ScientificFailure(ValueError):
    def __init__(self, field, value, reason, *, stage="REQUIRED_SCIENTIFIC_VALIDATION"):
        self.diagnostic = {
            "category": "REQUIRED_METADATA_CONTRACT_FAILURE"
            if stage == "REQUIRED_METADATA_VALIDATION"
            else "REQUIRED_SCIENTIFIC_VALIDATION_FAILURE",
            "field": field,
            "field_type": safe_type(value),
            "reason_code": reason,
            "required": True,
            "stage": stage,
            "exception_class": "ScientificFailure",
        }
        super().__init__(self.diagnostic["category"])


def required_metadata(raw, *, rows, features):
    """Local constructor/classes establish identity; remote assertions cannot contradict it.

    Missing remote assertions are UNKNOWN. Unsupported optional fields are handled later.
    A present malformed root/configuration, or contradicting scientific field, is fatal.
    """
    if not isinstance(raw, dict):
        raise ScientificFailure(
            "metadata", raw, "INVALID_CONTAINER", stage="REQUIRED_METADATA_VALIDATION"
        )
    expected = {"test_set_num_rows": rows, "test_set_num_cols": features, "classes": [0, 1]}
    result = {}
    for name, value in expected.items():
        actual = raw.get(name)
        if actual is None:
            result[name] = "UNKNOWN"
            continue
        if name == "classes":
            ok = isinstance(actual, (list, tuple, np.ndarray))
            if ok:
                a = np.asarray(actual)
                ok = a.shape == (2,) and a.tolist() in ([0, 1], ["0", "1"])
        else:
            ok = type(actual) is int and actual == value
        if not ok:
            raise ScientificFailure(
                name, actual, "CONTRACT_MISMATCH", stage="REQUIRED_METADATA_VALIDATION"
            )
        result[name] = value
    config = raw.get("tabpfn_config")
    if config is not None and not isinstance(config, dict):
        raise ScientificFailure(
            "tabpfn_config", config, "INVALID_CONTAINER", stage="REQUIRED_METADATA_VALIDATION"
        )
    for name, value in {
        **CONSTRUCTOR,
        "balance_probabilities": False,
        "ignore_pretraining_limits": False,
    }.items():
        actual = (config or {}).get(name)
        if actual is not None and (type(actual) is not type(value) or actual != value):
            raise ScientificFailure(
                name, actual, "CONTRACT_MISMATCH", stage="REQUIRED_METADATA_VALIDATION"
            )
        result[name] = value if actual is not None else "UNKNOWN"
    mode = raw.get("execution_mode")
    if mode is not None and (not isinstance(mode, str) or mode not in {"standard", "cache"}):
        raise ScientificFailure(
            "execution_mode", mode, "UNAPPROVED_RUNTIME_MODE", stage="REQUIRED_METADATA_VALIDATION"
        )
    result["execution_mode"] = mode if mode is not None else "UNKNOWN"
    return result


def optional_metadata(raw):
    """An explicitly unsupported optional representation becomes UNKNOWN, never raw text."""
    validators = {
        "billing_model_version": lambda v: (
            isinstance(v, str) and v in {"v3.5", "v3.5_default", "v3.5-plus", "3.5"}
        ),
        "cache_outcome": lambda v: (
            isinstance(v, str) and v in {"not_requested", "hit", "miss", "fallback"}
        ),
        "package_version": lambda v: (
            isinstance(v, str) and bool(re.fullmatch(r"\d+(?:\.\d+){1,3}(?:[a-z0-9.+-]{0,24})", v))
        ),
        "n_estimators": lambda v: type(v) is int and 0 < v < 100000,
    }
    safe, diagnostics = {}, []
    for name, validate in validators.items():
        value = raw.get(name)
        safe[name] = "UNKNOWN"
        if value is None:
            continue
        if validate(value):
            safe[name] = value
        else:
            diagnostics.append(
                {
                    "category": "OPTIONAL_METADATA_UNSUPPORTED_REPRESENTATION",
                    "field": name,
                    "field_type": safe_type(value),
                    "reason_code": "UNSUPPORTED_TYPE_OR_ENUM",
                    "required": False,
                    "stage": "OPTIONAL_METADATA_PROCESSING",
                    "exception_class": None,
                }
            )
    # Unknown keys, nested values, server IDs, response bodies and URLs are never retained.
    return {"fields": safe, "diagnostics": diagnostics, "unknown_fields_discarded": True}


def validate_probabilities(probabilities, classes, contract, observed):
    if observed != contract:
        raise ScientificFailure("input_contract", observed, "FROZEN_CONTRACT_MISMATCH")
    if contract["positive_class"] != 1 or contract["probability_column"] != 1:
        raise ScientificFailure("positive_class", contract, "SEMANTICS_MISMATCH")
    if contract["constructor"] != CONSTRUCTOR and contract["kind"] == "tabpfn":
        raise ScientificFailure("constructor", contract, "MODEL_MISMATCH")
    a = np.asarray(classes)
    if a.ndim != 1 or a.tolist() != [0, 1]:
        raise ScientificFailure("classes", classes, "ORDERED_BINARY_CLASSES_REQUIRED")
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (contract["query_rows"], 2):
        raise ScientificFailure("probability_shape", p, "BINARY_SHAPE_MISMATCH")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ScientificFailure("probabilities", p, "NONFINITE_OR_OUT_OF_RANGE")
    if not np.allclose(p.sum(axis=1), 1, atol=1e-6, rtol=0):
        raise ScientificFailure("probabilities", p, "ROW_SUM_MISMATCH")
    if len(contract["features"]) != contract["feature_count"]:
        raise ScientificFailure("features", contract, "FEATURE_COUNT_MISMATCH")
    if not contract["metadata_excluded_from_X"] or not contract["query_order_verified"]:
        raise ScientificFailure("query_order", contract, "INPUT_DISCIPLINE_FAILED")
    return p.copy()


def persist_result(path, probabilities, classes, contract, observed, raw_metadata, *, mark=None):
    mark = mark or (lambda stage: None)
    mark("PROBABILITY_VALIDATION_STARTED")
    p = validate_probabilities(probabilities, classes, contract, observed)
    required = required_metadata(raw_metadata, rows=len(p), features=contract["feature_count"])
    mark("PROBABILITY_VALIDATION_PASSED")
    path = Path(path)
    if path.exists():
        raise FileExistsError("Immutable probability checkpoint exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(
        path.parent / (path.name + ".reservation.json"),
        {"status": "RESERVED", "scientific_contract_sha256": digest(contract)},
    )
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=path.parent))
    try:
        with (temporary / "probabilities.npy").open("xb") as stream:
            np.save(stream, p, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        atomic_json(
            temporary / "manifest.json",
            {
                "status": "SCIENTIFICALLY_VALIDATED",
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "contract": contract,
                "required_metadata": required,
                "scientific_validation": "PASS",
                "positive_class_semantics": SEMANTICS,
                "probabilities_sha256": file_hash(temporary / "probabilities.npy"),
                "remote_order_assurance": "SDK positional contract; frozen query and export hashes",
            },
        )
        for file in temporary.iterdir():
            file.chmod(0o444)
        # Existing nonempty checkpoint cannot be replaced by POSIX rename.
        if path.exists():
            raise FileExistsError("Checkpoint destination appeared during publication")
        os.rename(temporary, path)
        path.chmod(0o555)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    mark("PROBABILITY_CHECKPOINT_PUBLISHED")
    # Only closed, optional unsupported representations continue. Unexpected adapter bugs
    # still stop publication, with the scientifically validated checkpoint preserved.
    optional = optional_metadata(raw_metadata)
    mark("OPTIONAL_METADATA_PROCESSED")
    return p[:, 1].copy(), optional
