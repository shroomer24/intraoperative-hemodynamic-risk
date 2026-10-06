"""Frozen study contract, development-only matrices, metadata and attempt guards."""

from study_bootstrap import REPO, STUDY  # isort: skip

import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from modeling_attempts import attempt_state
from modeling_process_support import atomic_bytes as atomic_bytes
from modeling_process_support import atomic_json

from intraop.data.modeling import DevelopmentPartitions, read_json, sha256
from intraop.evaluation.development import development_metrics

SPEC_HASH = "4905031dae7ba13497df7e72564c31ced350c884af8758014b617ebca4a26a13"
IDS = (
    "A_frozen74_retained_control",
    "B_structural71",
    "C_causal_contrasts78",
    "D_positive_preserving_negative_thinning74",
)
AUDIT_PLAN = REPO / "artifacts/tabpfn-optimization-audit-v01/bounded_experiment_plan.json"
ORIGINAL = REPO / "artifacts/modeling-v01"
BASELINE = ORIGINAL / "development/candidates/tabpfn_full__v3p5_default_seed42.csv"
CONSTRUCTOR = {"model_path": "v3.5_default", "random_state": 42}
DERIVED = (
    ("map_recent_minus_oldest_minute", "map_minute_5_median", "map_minute_1_median"),
    ("map_short_minus_long_slope", "map_slope_60s", "map_slope_300s"),
    ("sbp_latest_minus_history_median", "sbp_latest", "sbp_median_300s"),
    ("dbp_latest_minus_history_median", "dbp_latest", "dbp_median_300s"),
)


def timestamp():
    return datetime.now(UTC).isoformat()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def positions_hash(positions):
    # Explicit representation: little-endian signed int64 positional indices, C order.
    return hashlib.sha256(np.asarray(positions, dtype="<i8").tobytes(order="C")).hexdigest()


def array_hash(array):
    a = np.ascontiguousarray(array, dtype="<f8")
    a[np.isnan(a)] = np.nan  # canonical NaN payload, no imputation
    return hashlib.sha256(a.tobytes()).hexdigest()


def verified_spec():
    if sha256(AUDIT_PLAN) != SPEC_HASH or sha256(STUDY / "study_plan.json") != SPEC_HASH:
        raise ValueError("Authoritative approved study specification changed")
    plan = read_json(AUDIT_PLAN)
    approval = read_json(STUDY / "architectural_approval.json")
    if approval.get("status") != "APPROVED" or approval.get("study_plan_sha256") != SPEC_HASH:
        raise PermissionError("Exact architectural approval missing")
    if approval.get("approved_new_variants") != list(IDS[1:]):
        raise PermissionError("Approved candidate set differs")
    if [v["id"] for v in plan["variants"]] != list(IDS):
        raise ValueError("Variant set/order differs")
    if plan["invariants"]["constructor"] != CONSTRUCTOR or plan["configurations_per_variant"] != 1:
        raise ValueError("Frozen constructor/configuration differs")
    if plan["allowed_partitions"] != ["training", "tuning"]:
        raise PermissionError("Development partition capability changed")
    if (ORIGINAL / "model_lock_manifest.json").exists():
        raise PermissionError("Study forbidden after final model lock")
    if sha256(ORIGINAL / "development_plan.json") != plan["source_development_plan_sha256"]:
        raise ValueError("Original development plan changed")
    if (
        sha256(ORIGINAL / "development/development_completion_manifest.json")
        != plan["source_development_completion_sha256"]
    ):
        raise ValueError("Completed development manifest changed")
    if sha256(BASELINE) != plan["variants"][0]["prediction_file_sha256"]:
        raise ValueError("Retained baseline A prediction hash changed")
    return plan


def environment(plan, *, external=False):
    inv = plan["invariants"]
    if (
        sys.executable != inv["python_executable"]
        or sys.version.split()[0] != inv["python_version"]
    ):
        raise RuntimeError("Exact pinned Python interpreter/version differs")
    expected = read_json(ORIGINAL / "development_plan.json")["package_versions"]
    actual = {n: importlib.metadata.version(n) for n in expected}
    if actual != expected or actual["tabpfn-client"] != "0.6.1":
        raise RuntimeError("Pinned scientific package versions differ")
    if external:
        if os.environ.get("TABPFN_STUDY_EXTERNAL_EXECUTION") != "normal_terminal":
            raise PermissionError("Use only the secure normal-terminal study wrapper")
        if not os.environ.get("TABPFN_TOKEN"):
            raise RuntimeError("TABPFN_TOKEN unavailable; credential values are never recorded")
    return {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "package_versions": actual,
        "credential_values_recorded": False,
    }


def load_development(plan):
    original = read_json(ORIGINAL / "development_plan.json")
    for name, digest in original["code_hashes"].items():
        if sha256(REPO / name) != digest:
            raise ValueError("Frozen scientific source changed")
    if sha256(ORIGINAL / "feature_sets.json") != plan["frozen_feature_sets_sha256"]:
        raise ValueError("Frozen feature contract changed")
    loader = DevelopmentPartitions(
        REPO / plan["frozen_checkpoint"], ORIGINAL, plan["frozen_checkpoint_manifest_hashes"]
    )
    train, tuning = loader.load("training"), loader.load("tuning")
    if (len(train.X), len(tuning.X), int(train.y.sum()), int(tuning.y.sum())) != (
        13975,
        3193,
        498,
        94,
    ):
        raise ValueError("Frozen partition row/class counts differ")
    if train.groups.nunique() != 90 or tuning.groups.nunique() != 23:
        raise ValueError("Frozen subject counts differ")
    if set(train.groups) & set(tuning.groups):
        raise ValueError("TRAINING/TUNING subjects overlap")
    for data in (train, tuning):
        if data.X.columns.tolist() != plan["variants"][0]["feature_names"]:
            raise ValueError("Original ordered 74-feature contract differs")
        if not data.X.index.is_unique or not data.X.index.is_monotonic_increasing:
            raise ValueError("Original source-row ordering differs")
        if not data.metadata.window_id.is_unique or data.y.dtype != np.dtype("int64"):
            raise ValueError("Frozen label dtype or window uniqueness differs")
    return loader, train, tuning


def variant_definition(plan, variant):
    if variant not in IDS[1:]:
        raise PermissionError("Only B, C and D may create a new hosted candidate")
    return next(v for v in plan["variants"] if v["id"] == variant)


def represented_features(data, plan, variant):
    definition = variant_definition(plan, variant)
    full = plan["variants"][0]["feature_names"]
    if data.X.columns.tolist() != full:
        raise ValueError("Input feature order differs")
    if variant == IDS[1]:
        expected = [
            n
            for n in full
            if n
            not in {"map_minute_5_median", "dbp_missing_fraction", "dbp_measurement_age_seconds"}
        ]
        frame = data.X.loc[:, expected].copy()
    elif variant == IDS[2]:
        frame = data.X.copy()
        for name, left, right in DERIVED:
            frame[name] = data.X[left] - data.X[right]
    else:
        frame = data.X.copy()
    if frame.columns.tolist() != definition["feature_names"]:
        raise ValueError("Approved feature order/arithmetic definition differs")
    if set(frame.columns) & set(data.metadata.columns):
        raise ValueError("Metadata cannot enter predictors")
    matrix = frame.to_numpy(dtype=np.float64, copy=True)
    if matrix.dtype != np.float64 or np.isinf(matrix).any():
        raise ValueError("Predictors must be float64 and finite-or-NaN")
    return matrix


def negative_thinning_positions(training):
    y = training.y.to_numpy(dtype=np.int64)
    meta = training.metadata
    if not meta.index.equals(training.X.index) or not training.y.index.equals(training.X.index):
        raise ValueError("TRAINING metadata/label alignment differs")
    if not np.isfinite(meta.anchor_time_seconds.to_numpy()).all():
        raise ValueError("TRAINING anchor timestamps are invalid")
    frame = meta.loc[:, ["case_id", "anchor_time_seconds"]].copy().reset_index(drop=True)
    frame["source_row"] = training.X.index.to_numpy()
    frame["position"] = np.arange(len(y), dtype=np.int64)
    frame["label"] = y
    selected = set(np.flatnonzero(y == 1).tolist())
    for _, group in frame.loc[frame.label == 0].groupby("case_id", sort=False):
        last = None
        group = group.sort_values(["anchor_time_seconds", "source_row"], kind="stable")
        for row in group.itertuples(index=False):
            if last is None or row.anchor_time_seconds - last >= 300:
                selected.add(int(row.position))
                last = row.anchor_time_seconds
    return np.asarray(sorted(selected), dtype=np.int64)


def context_manifest(training, positions):
    y = training.y.to_numpy(dtype=np.int64)
    positive = np.flatnonzero(y == 1)
    chosen = training.metadata.iloc[positions]
    retained_y = y[positions]
    all_positive = bool(np.isin(positive, positions).all())
    result = {
        "variant": IDS[3],
        "scope": "TRAINING fitting context only",
        "spacing_seconds": 300,
        "positive_resets_negative_clock": False,
        "retained_total_rows": len(positions),
        "retained_positives": int(retained_y.sum()),
        "retained_negatives": int((retained_y == 0).sum()),
        "context_prevalence": float(retained_y.mean()),
        "retained_subjects": int(chosen.subject_id.nunique()),
        "represented_positive_subjects": int(chosen.loc[retained_y == 1].subject_id.nunique()),
        "represented_positive_episodes": int(
            chosen.loc[retained_y == 1]
            .groupby(["case_id", "matched_episode_onset_seconds"])
            .ngroups
        )
        if "matched_episode_onset_seconds" in chosen
        else None,
        "selected_position_sha256": positions_hash(positions),
        "position_encoding": "zero-based TRAINING positions, little-endian int64 C-order bytes",
        "selected_source_row_sha256": positions_hash(training.X.index.to_numpy()[positions]),
        "all_original_positive_rows_retained": all_positive,
        "original_positive_rows": int(y.sum()),
        "positive_source_row_sha256": positions_hash(training.X.index.to_numpy()[positive]),
        "context_restored_to_original_source_order": bool(np.all(np.diff(positions) > 0)),
        "metadata_transmitted_in_X": False,
    }
    if not all_positive or set(chosen.subject_id) != set(training.groups):
        raise ValueError("D lost original positives or subjects; do not adjust the rule")
    if (
        result["represented_positive_subjects"]
        != training.metadata.loc[y == 1].subject_id.nunique()
    ):
        raise ValueError("D lost a positive subject")
    return result


def matrices(training, tuning, plan, variant):
    X = represented_features(training, plan, variant)
    query = represented_features(tuning, plan, variant)
    positions = negative_thinning_positions(training) if variant == IDS[3] else np.arange(len(X))
    y = training.y.to_numpy(dtype=np.int64)[positions]
    if np.unique(y).tolist() != [0, 1]:
        raise ValueError("Context must contain exactly label classes [0,1]")
    if len(query) != len(tuning.X):
        raise ValueError("Complete TUNING population must remain unchanged")
    return X[positions], y, query, positions


def validate_predictions(path, tuning, *, variant=None):
    rows = pd.read_csv(path, dtype={n: str for n in ["subject_id", "case_id", "window_id"]})
    if len(rows) != len(tuning.X):
        raise ValueError("Prediction row count differs")
    for name, values in [
        ("source_row", tuning.X.index.to_numpy()),
        ("true_label", tuning.y.to_numpy()),
        ("subject_id", tuning.metadata.subject_id.to_numpy()),
        ("case_id", tuning.metadata.case_id.to_numpy()),
        ("window_id", tuning.metadata.window_id.to_numpy()),
        ("anchor_time_seconds", tuning.metadata.anchor_time_seconds.to_numpy()),
    ]:
        if not np.array_equal(rows[name].to_numpy(), values):
            raise ValueError("Prediction row alignment differs")
    if set(rows.split) != {"tuning"}:
        raise PermissionError("Predictions must be TUNING only")
    if variant and (
        set(rows.candidate_id) != {variant}
        or set(rows.model_identifier) != {"tabpfn_representation_study"}
    ):
        raise ValueError("Candidate prediction identity differs")
    p = rows.predicted_probability.to_numpy(dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Invalid class-1 probabilities")
    return rows, development_metrics(tuning.y, p, probability_output=True)


def class_one_probability(probabilities, classes, rows):
    if np.asarray(classes).tolist() != [0, 1]:
        raise ValueError("Local classes must equal [0,1]")
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (rows, 2) or not np.isfinite(probabilities).all():
        raise ValueError("Binary probability shape/finite contract differs")
    if (
        (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    ):
        raise ValueError("Probability range/normalization differs")
    return probabilities[:, 1].copy()


ENUMS = {
    "model_path": {"v3.5_default"},
    "inference_precision": {
        "auto",
        "autocast",
        "float32",
        "float64",
        "float16",
        "bfloat16",
        "torch.float32",
        "torch.float64",
        "torch.float16",
        "torch.bfloat16",
    },
    "fit_mode": {"fit_preprocessors", "fit_with_cache"},
    "FEATURE_SHIFT_METHOD": {"shuffle", "rotate", "none"},
    "CLASS_SHIFT_METHOD": {"shuffle", "rotate", "none"},
    "SAMPLE_SUBSAMPLING_METHOD": {
        "auto",
        "random",
        "stratified",
        "majority_downsample",
        "balanced",
    },
    "FEATURE_SUBSAMPLING_METHOD": {"random", "balanced"},
    "POLYNOMIAL_FEATURES": {"no", "all"},
    "OUTLIER_REMOVAL_STD": {"auto"},
    "name": {
        "none",
        "safepower",
        "quantile_uni",
        "quantile_norm",
        "robust",
        "squashing_scaler_default",
        "quantile_uni_extrapolate",
    },
    "categorical_name": {"ordinal", "ordinal_shuffled", "none", "onehot"},
    "global_transformer_name": {"svd", "none"},
}
NUMERIC = {
    "n_estimators",
    "random_state",
    "softmax_temperature",
    "SUBSAMPLE_SAMPLES",
    "OUTLIER_REMOVAL_STD",
    "N_ESTIMATORS",
    "SOFTMAX_TEMPERATURE",
    "MAX_NUMBER_OF_SAMPLES",
    "MAX_NUMBER_OF_FEATURES",
    "MAX_NUMBER_OF_CLASSES",
    "MAX_UNIQUE_FOR_CATEGORICAL_FEATURES",
    "MIN_UNIQUE_FOR_NUMERICAL_FEATURES",
    "max_features_per_estimator",
    "TEXT_N_COMPONENTS",
    "MIN_CARDINALITY_FOR_TEXT",
}
BOOLEANS = {
    "balance_probabilities",
    "average_before_softmax",
    "ignore_pretraining_limits",
    "FINGERPRINT_FEATURE",
    "ENABLE_GPU_PREPROCESSING",
    "PASSTHROUGH_INF",
    "TRANSFORM_TEXT",
    "TRANSFORM_DATES",
    "append_original",
    "USE_SKLEARN_16_DECIMAL_PRECISION",
}


def safe_config(config):
    if not isinstance(config, dict):
        raise ValueError("Resolved configuration has unexpected structure")
    safe = {}
    for key, value in config.items():
        if (
            key not in ENUMS
            and key not in NUMERIC
            and key not in BOOLEANS
            and key
            not in {"inference_config", "PREPROCESS_TRANSFORMS", "categorical_features_indices"}
        ):
            continue
        if value is None:
            safe[key] = None
        elif key in ENUMS and isinstance(value, str) and value in ENUMS[key]:
            safe[key] = value
        elif key in BOOLEANS and isinstance(value, bool):
            safe[key] = value
        elif (
            key in NUMERIC
            and not isinstance(value, bool)
            and isinstance(value, (int, float))
            and np.isfinite(value)
        ):
            safe[key] = value
        elif key == "inference_config":
            safe[key] = safe_config(value)
        elif (
            key == "PREPROCESS_TRANSFORMS"
            and isinstance(value, list)
            and all(isinstance(v, dict) for v in value)
        ):
            safe[key] = [safe_config(v) for v in value]
        elif (
            key == "categorical_features_indices"
            and isinstance(value, list)
            and all(type(v) is int and v >= 0 for v in value)
        ):
            safe[key] = value
        else:
            raise ValueError("Known resolved configuration field failed safe type/enum validation")
    return safe


def safe_metadata(raw, *, query_rows, feature_count):
    if not isinstance(raw, dict):
        raise ValueError("Remote metadata must be a mapping")
    safe = {}
    for key in ["n_estimators", "test_set_num_rows", "test_set_num_cols"]:
        if key in raw and raw[key] is not None:
            if type(raw[key]) is not int or raw[key] < 1:
                raise ValueError("Remote count metadata invalid")
            safe[key] = raw[key]
    for key, choices in [
        ("billing_model_version", {"v3.5", "v3.5_default", "v3.5-plus", "3.5"}),
        ("execution_mode", {"standard", "thinking", "cache"}),
        ("cache_outcome", {"not_requested", "hit", "miss", "fallback"}),
    ]:
        if key in raw and raw[key] is not None:
            if raw[key] not in choices:
                raise ValueError("Unexpected remote model/runtime enum")
            safe[key] = raw[key]
    if safe.get("execution_mode") == "thinking":
        raise ValueError("Unexpected Thinking execution for frozen standard constructor")
    if "package_version" in raw and raw["package_version"] is not None:
        value = raw["package_version"]
        if not isinstance(value, str) or not re.fullmatch(
            r"\d+(?:\.\d+){1,3}(?:[a-z0-9.+-]{0,24})", value
        ):
            raise ValueError("Remote package version is not a safe version identifier")
        safe["package_version"] = value
    if "classes" in raw and raw["classes"] is not None:
        classes = raw["classes"]
        if classes not in ([0, 1], ["0", "1"]):
            raise ValueError("Remote classes differ from ordered [0,1]")
        safe["classes"] = [0, 1]
    if "tabpfn_config" in raw and raw["tabpfn_config"] is not None:
        safe["tabpfn_config"] = safe_config(raw["tabpfn_config"])
        for key, expected in {
            **CONSTRUCTOR,
            "balance_probabilities": False,
            "ignore_pretraining_limits": False,
        }.items():
            if key in safe["tabpfn_config"] and safe["tabpfn_config"][key] != expected:
                raise ValueError("Remote resolved configuration contradicts frozen constructor")
    if (
        safe.get("test_set_num_rows", query_rows) != query_rows
        or safe.get("test_set_num_cols", feature_count) != feature_count
    ):
        raise ValueError("Remote query dimensions disagree with submitted matrix")
    return {
        "available": bool(safe),
        "metadata": safe,
        "unavailable_fields": [
            n
            for n in [
                "billing_model_version",
                "package_version",
                "n_estimators",
                "execution_mode",
                "cache_outcome",
                "classes",
                "test_set_num_rows",
                "test_set_num_cols",
                "tabpfn_config",
            ]
            if n not in safe
        ],
        "historical_baseline_resolved_metadata": "UNKNOWN",
        "unknown_fields_discarded": True,
        "credentials_or_patient_payloads_retained": False,
    }


def runtime_signature(report):
    # Cache outcome and query dimensions are operational/data-dependent, not configuration drift.
    meta = report["metadata"]
    config = {
        k: v
        for k, v in meta.get("tabpfn_config", {}).items()
        if k != "categorical_features_indices"
    }
    return {
        n: meta.get(n, "UNKNOWN")
        for n in ["billing_model_version", "package_version", "n_estimators", "execution_mode"]
    } | {"tabpfn_config": config}


def require_consistent_metadata(reports):
    signatures = [runtime_signature(r) for r in reports]
    if signatures and any(s != signatures[0] for s in signatures[1:]):
        raise RuntimeError(
            "Remote runtime/configuration differs across candidates; review required"
        )
    return {
        "status": "NO_OBSERVED_RUNTIME_CONFIGURATION_DIFFERENCE",
        "available_signature": signatures[0] if signatures else {},
        "historical_baseline_comparability": "UNKNOWN",
        "excluded_operational_fields": [
            "cache_outcome",
            "test_set_num_rows",
            "test_set_num_cols",
            "categorical_features_indices",
        ],
    }


def inspect_processes():
    try:
        result = subprocess.run(
            ["/bin/ps", "-axo", "pid=,comm=,args="], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        raise RuntimeError("Process inspection unavailable; hosted launch blocked") from None
    relevant = []
    script_names = {
        "study_worker.py",
        "run_study.py",
        "modeling_tabpfn_worker.py",
        "modeling_development_resume.py",
    }
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) != 3:
            continue
        pid, exe, command = fields
        if int(pid) == os.getpid() or not Path(exe).name.lower().startswith("python"):
            continue
        try:
            words = shlex.split(command)
        except ValueError:
            raise RuntimeError("Unparseable Python process; hosted launch blocked") from None
        if {Path(w).name for w in words} & script_names:
            relevant.append(int(pid))
    return relevant


def require_inactive(scanner=None):
    if (scanner or inspect_processes)():
        raise RuntimeError("ACTIVE worker/parent; duplicate hosted request blocked")


def candidate_state(directory, *, active=False):
    bundle = directory / "bundle"
    logs = directory / "attempt_1/logs"
    state = attempt_state(bundle, directory / "prediction.csv", logs, active=active)
    if state == "NEW" and directory.exists():
        return "INDETERMINATE"
    return state


def reserve_attempt(directory, identity, *, scanner=None):
    require_inactive(scanner)
    state = candidate_state(directory)
    if state != "NEW":
        raise RuntimeError(f"Candidate state {state}; no automatic duplicate evaluation or retry")
    atomic_json(
        directory / "attempt_1/reservation.json",
        {
            "status": "reserved",
            "attempt_number": 1,
            "scientific_identity_sha256": identity,
            "study_plan_sha256": SPEC_HASH,
            "reserved_utc": timestamp(),
            "independent_candidate_evaluation_number": 1,
            "automatic_retry_permitted": False,
        },
        exclusive=True,
    )
    return directory / "attempt_1/logs"


def verify_preparation(plan):
    manifest = read_json(STUDY / "preparation_manifest.json")
    if (
        sha256(STUDY / "preparation_manifest.json")
        != (STUDY / "preparation_manifest.sha256").read_text().strip()
    ):
        raise ValueError("Preparation manifest hash differs")
    if manifest["study_plan_sha256"] != SPEC_HASH:
        raise ValueError("Preparation specification differs")
    for name, digest in manifest["code_hashes"].items():
        if sha256(REPO / name) != digest:
            raise ValueError("Prepared implementation changed")
    for name, digest in manifest["prepared_artifacts_sha256"].items():
        if sha256(STUDY / name) != digest:
            raise ValueError("Prepared input evidence changed")
    return manifest


def selected_representation(records):
    tolerance = 1e-12
    if {r["variant"] for r in records} != set(IDS):
        raise ValueError("Selection requires exactly A/B/C/D")
    a = next(r for r in records if r["variant"] == IDS[0])
    best_ap = max(r["average_precision"] for r in records)
    if best_ap <= a["average_precision"] + tolerance:
        return a
    contenders = [r for r in records if best_ap - r["average_precision"] <= tolerance]
    best_loss = min(r["log_loss"] for r in contenders)
    contenders = [r for r in contenders if r["log_loss"] <= best_loss + tolerance]
    return min(contenders, key=lambda r: (IDS.index(r["variant"]), r["variant"]))


def paired_bootstrap(tuning, probabilities):
    y = tuning.y.to_numpy(dtype=np.int64)
    groups = list(tuning.metadata.groupby("subject_id", sort=False).indices.values())
    rng = np.random.default_rng(20261003)
    differences = {v: [] for v in IDS[1:]}
    skipped = 0
    from sklearn.metrics import average_precision_score

    for _ in range(1000):
        indices = np.concatenate(
            [groups[i] for i in rng.integers(0, len(groups), size=len(groups))]
        )
        if np.unique(y[indices]).tolist() != [0, 1]:
            skipped += 1
            continue
        baseline = average_precision_score(y[indices], probabilities[IDS[0]][indices])
        for variant in IDS[1:]:
            differences[variant].append(
                float(
                    average_precision_score(y[indices], probabilities[variant][indices]) - baseline
                )
            )
    return {
        "seed": 20261003,
        "requested_replicates": 1000,
        "valid_replicates": 1000 - skipped,
        "one_class_replicates_skipped": skipped,
        "unit": "TUNING subject with all its windows",
        "descriptive_only": True,
        "used_for_selection": False,
        "AP_difference_95pct_percentile": {
            v: np.quantile(d, [0.025, 0.975]).tolist() for v, d in differences.items()
        },
    }
