"""One final external TEST run; no overrides, resume, refitting or interpretation."""

import json
import resource
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import final_test_reporting as reporting  # noqa: E402
import final_test_support as s  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def worker_sources(query, registry, ph):
    raw, identities = {}, {}
    order = s.a.row_hash(query.X.index)
    for name in s.MODELS:
        worker = s.ROOT / "workers" / name
        receipt, _ = s.a.verify_seal(worker / "completion_receipt.json")
        expected_fits = 1 if name in s.HOSTED else 0
        expected_predicts = 0 if name == "current_map" else 1
        actual = s.a.receipt_files(worker)
        snapshot = {
            k: v
            for k, v in actual.items()
            if k not in ["completion_receipt.json", "completion_receipt.sha256"]
        }
        if (
            receipt["status"] != "COMPLETE"
            or receipt["phase"] != "test"
            or receipt["model"] != name
            or receipt["execution_preparation_sha256"] != ph
            or receipt["source_registry_sha256"] != s.REGISTRY_HASH
            or receipt["partition_access"] != ["training", "test"]
            or receipt["fit_calls"] != expected_fits
            or receipt["predict_proba_calls"] != expected_predicts
            or receipt["Platt_fit_calls"] != 0
            or receipt["automatic_retry"]
            or receipt["files_sha256"] != snapshot
            or (worker / "failure.json").exists()
        ):
            raise PermissionError("Incomplete/changed TEST worker evidence")
        score = name == "current_map"
        cp = worker / ("score_checkpoint" if score else "probability_checkpoint")
        manifest, mh = s.a.verify_seal(cp / "manifest.json")
        contract = manifest["contract"]
        record = registry["sources"][name]
        if (
            manifest["scientific_validation"] != "PASS"
            or contract["query_source_order_sha256"] != order
            or contract["query_rows"] != len(query.X)
            or contract["phase"] != "test"
            or contract["model_lock_sha256"] != s.a.LOCK_HASH
            or contract["feature_contract_sha256"] != record["feature_contract_sha256"]
            or contract["features"] != record["features"]
            or contract["training_X_sha256"] != record["training_X_sha256"]
            or contract["training_y_sha256"] != record["training_y_sha256"]
            or contract["model_definition_sha256"] != record["locked_definition_sha256"]
            or contract["query_X_sha256"]
            != s.a.array_hash(s.a.full_support.predictor_array(query, record["features"]))
            or not contract["query_order_verified"]
            or not contract["metadata_excluded_from_X"]
        ):
            raise PermissionError("Frozen TEST order/feature/training/model contract differs")
        path = cp / ("scores.npy" if score else "probabilities.npy")
        h = s.a.file_hash(path)
        if h != manifest["scores_sha256" if score else "probabilities_sha256"]:
            raise PermissionError("Immutable TEST result hash differs")
        arr = np.load(path, allow_pickle=False)
        if score:
            if manifest["score_definition"] != "-map_latest" or manifest["probability_mapping"]:
                raise PermissionError("Current-MAP score cannot be transformed")
            if not np.array_equal(arr, -query.X.map_latest.to_numpy(dtype=float)):
                raise PermissionError("Current-MAP score differs from frozen TEST input")
            raw[name] = arr
            identity = {"kind": "current_map", "score_definition": "-map_latest"}
        else:
            s.a.full_support.validate_probabilities(arr, manifest["classes"], contract, contract)
            raw[name] = arr[:, 1].copy()
            if name in s.HOSTED:
                required = manifest["required_metadata"]
                if (
                    required["model_path"] not in ("v3.5_default", s.a.full_support.CANONICAL)
                    or required["billing_model_version"] != "v3.5"
                    or required["execution_mode"] != "standard"
                ):
                    raise PermissionError("Exact approved hosted identity required")
                identity = {
                    "requested_constructor": s.a.CONSTRUCTOR,
                    "returned_model_path": required["model_path"],
                    "billing_model_version": required["billing_model_version"],
                    "execution_mode": required["execution_mode"],
                    "classes": [0, 1],
                }
            else:
                identity = {
                    "kind": record["kind"],
                    "fitted_model_sha256": record["fitted_model_sha256"],
                    "classes": [0, 1],
                }
        if raw[name].shape != (len(query.X),) or not np.isfinite(raw[name]).all():
            raise ValueError("TEST result row/numeric contract differs")
        if name == "prevalence" and not np.allclose(
            raw[name], registry["fixed_training_prevalence"], atol=1e-15, rtol=0
        ):
            raise PermissionError("Fixed TRAINING prevalence changed")
        identities[name] = {
            **identity,
            "locked_definition_sha256": record["locked_definition_sha256"],
            "feature_contract_sha256": record["feature_contract_sha256"],
            "query_source_order_sha256": order,
            "checkpoint_manifest_sha256": mh,
            "result_sha256": h,
            "checkpoint_path": str(cp.relative_to(s.REPO)),
        }
    return raw, identities


def validate_publication(staged, query, raw, mappings, identities):
    expected = {
        "source_registry.json",
        "source_registry.sha256",
        "bootstrap_draws.npz",
        "reliability.png",
        *[f"predictions/{name}.{suffix}" for name in s.MODELS for suffix in ["csv", "npz"]],
        *[
            f"{name}.{suffix}"
            for name in [
                "metrics",
                "reliability",
                "subject_bootstrap",
                "alignment",
                "model_identities",
                "environment_provenance",
            ]
            for suffix in ["json", "sha256"]
        ],
    }
    files = s.a.receipt_files(staged)
    if set(files) != expected or any(
        p.stat().st_mode & 0o222 for p in staged.rglob("*") if p.is_file()
    ):
        raise PermissionError("Exact read-only publication set required")
    s.a.verify_seal(staged / "source_registry.json", s.REGISTRY_HASH)
    alignment, _ = s.a.verify_seal(staged / "alignment.json")
    y = query.y.to_numpy(dtype=int)
    if (
        alignment["query_source_order_sha256"] != s.a.row_hash(query.X.index)
        or alignment["labels_sha256"] != s.a.array_hash(y)
        or alignment["rows"] != len(y)
        or alignment["subjects"] != query.groups.nunique()
        or alignment["positive_labels"] != int(y.sum())
        or alignment["patient_identifiers_published"]
    ):
        raise ValueError("TEST label/ordinal alignment differs")
    if s.a.verify_seal(staged / "model_identities.json")[0] != identities:
        raise PermissionError("Published model identities differ")
    expected_metrics, expected_tables = {}, {}
    for name in s.MODELS:
        calibrated = (
            reporting.apply_mapping(raw[name], mappings[name]) if name in s.LEARNED else None
        )
        key = "raw_score" if name == "current_map" else "raw_probability"
        values = {"query_position": np.arange(len(y)), "true_label": y, key: raw[name]}
        if calibrated is not None:
            values["calibrated_probability"] = calibrated
        with np.load(staged / "predictions" / f"{name}.npz", allow_pickle=False) as arrays:
            if set(arrays.files) != set(values) or any(
                not np.array_equal(arrays[k], v) for k, v in values.items()
            ):
                raise ValueError("Prediction arrays differ or contain identifiers")
        frame = pd.read_csv(staged / "predictions" / f"{name}.csv", float_precision="round_trip")
        if frame.columns.tolist() != [
            "query_position",
            "true_label",
            "raw_probability",
            "raw_score",
            "calibrated_probability",
            "model",
            "split",
        ]:
            raise ValueError("Prediction CSV schema differs or exposes identifiers")
        if (
            not np.array_equal(frame[key], raw[name])
            or not np.array_equal(frame.true_label, y)
            or not np.array_equal(frame.query_position, np.arange(len(y)))
            or not frame.model.eq(name).all()
            or not frame.split.eq("test").all()
        ):
            raise ValueError("Prediction CSV values/order differ")
        unused = "raw_probability" if name == "current_map" else "raw_score"
        if (
            not frame[unused].isna().all()
            or (calibrated is None and not frame.calibrated_probability.isna().all())
            or (
                calibrated is not None
                and not np.array_equal(frame.calibrated_probability, calibrated)
            )
        ):
            raise ValueError("Probability/score or calibration discipline differs")
        probability = name != "current_map"
        m = reporting.development_metrics(y, raw[name], probability_output=probability)
        expected_metrics[name + "/raw"] = (
            m
            if probability
            else {
                k: m[k]
                for k in [
                    "windows",
                    "positives",
                    "average_precision",
                    "auroc",
                    "probability_output",
                ]
            }
        )
        if probability:
            expected_tables[name + "/raw"] = reporting.reliability(y, raw[name])
        if calibrated is not None:
            expected_metrics[name + "/calibrated"] = reporting.development_metrics(
                y, calibrated, probability_output=True
            )
            expected_tables[name + "/calibrated"] = reporting.reliability(y, calibrated)
    if (
        s.a.verify_seal(staged / "metrics.json")[0] != expected_metrics
        or s.a.verify_seal(staged / "reliability.json")[0] != expected_tables
    ):
        raise ValueError("Prespecified diagnostic outputs differ")
    with np.load(staged / "bootstrap_draws.npz", allow_pickle=False) as arrays:
        _, codes = np.unique(query.groups.to_numpy(), return_inverse=True)
        draws = np.random.default_rng(42).integers(
            0, query.groups.nunique(), size=(1000, query.groups.nunique())
        )
        defined = np.array(
            [
                np.unique(y[np.concatenate([np.flatnonzero(codes == i) for i in draw])]).size == 2
                for draw in draws
            ]
        )
        if (
            set(arrays.files)
            != {"subject_ordinal_draws", "query_subject_ordinal", "replicate_defined"}
            or not np.array_equal(arrays["subject_ordinal_draws"], draws)
            or not np.array_equal(arrays["query_subject_ordinal"], codes)
            or not np.array_equal(arrays["replicate_defined"], defined)
        ):
            raise ValueError("Exact shared subject bootstrap draws differ")
    boot, _ = s.a.verify_seal(staged / "subject_bootstrap.json")
    if (
        boot["protocol"] != s.TEST_PROTOCOL
        or boot["draws_sha256"] != s.a.file_hash(staged / "bootstrap_draws.npz")
        or not boot["shared_draws_all_models"]
        or set(boot["intervals"]) != set(expected_metrics)
    ):
        raise ValueError("Fixed bootstrap protocol/output set differs")
    for row in boot["intervals"].values():
        if set(row) != {"average_precision", "auroc"}:
            raise ValueError("AP/AUROC bootstrap required")
        for result in row.values():
            if result["valid_replicates"] != int(defined.sum()) or result[
                "undefined_replicates"
            ] != 1000 - int(defined.sum()):
                raise ValueError("Undefined bootstrap accounting differs")
    if (staged / "reliability.png").read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Reliability figure missing")
    return files


def execute():
    prep, ph, lock, registry, mappings = s.reserve()
    active_name, code, stage = None, None, "RESERVED"
    try:
        for name in s.MODELS:
            active_name, code = name, None
            stage = "WORKER_LAUNCHED"
            print(json.dumps({"status": "STARTING", "phase": "test", "model": name}), flush=True)
            result = subprocess.run(
                [s.a.PYTHON, "-I", "-u", str(s.BASE / "final_test_worker.py"), "--model", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            code = result.returncode
            s.a.atomic_json(
                s.ROOT / "process_logs" / f"{name}.json",
                {
                    "exit_code": code,
                    "signal": -code if code < 0 else None,
                    "raw_streams_retained": False,
                    "fresh_exec": True,
                },
            )
            if code:
                raise RuntimeError("TEST worker stopped; no retry")
        active_name, code, stage = None, None, "WORKERS_COMPLETE"
        loader = s.TestPartitions()
        query = loader.load("test")
        raw, identities = worker_sources(query, registry, ph)
        stage = "SOURCES_VALIDATED"
        staged = reporting.publish(query, raw, mappings, identities, registry, prep, ph)
        stage = "OUTPUTS_STAGED"
        files = validate_publication(staged, query, raw, mappings, identities)
        stage = "OUTPUTS_VERIFIED"
        loader.authorize()
        for relative in sorted(files):
            s.a.copy_immutable(staged / relative, s.ROOT / relative)
        if {rel: s.a.file_hash(s.ROOT / rel) for rel in files} != files:
            raise PermissionError("Final publication hashes differ")
        loader.authorize()
        stage = "OUTPUTS_PUBLISHED"
        s.a.seal_json(
            s.RECEIPT,
            {
                "status": "COMPLETE",
                "phase": "test",
                "models": list(s.MODELS),
                "execution_count": 1,
                "model_lock_sha256": s.a.LOCK_HASH,
                "test_protocol_sha256": s.TEST_PROTOCOL_HASH,
                "calibration_completion_receipt_sha256": s.CALIBRATION_RECEIPT_HASH,
                "calibration_parameters_sha256": s.MAPPINGS_HASH,
                "source_registry_sha256": s.REGISTRY_HASH,
                "execution_preparation_sha256": ph,
                "execution_source_sha256": prep["bound_files_sha256"],
                "rows": len(query.X),
                "subjects": query.groups.nunique(),
                "positive_labels": int(query.y.sum()),
                "query_source_order_sha256": s.a.row_hash(query.X.index),
                "files_sha256": s.a.receipt_files(s.ROOT),
                "final_artifacts_sha256": files,
                "model_fit_calls": {name: 1 if name in s.HOSTED else 0 for name in s.MODELS},
                "Platt_refit_calls": 0,
                "training_context": "frozen TRAINING only",
                "automatic_retry": False,
                "configuration_changes_allowed": False,
                "interpretation_embedded": False,
                "timestamp_utc": s.a.stamp(),
            },
        )
        stage = "RECEIPT_CREATED"
        receipt, _ = s.a.verify_seal(s.RECEIPT)
        for rel, h in receipt["files_sha256"].items():
            if s.a.file_hash(s.ROOT / rel) != h:
                raise PermissionError("Completion receipt verification failed")
        for path in sorted(s.ROOT.rglob("*"), reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
        s.ROOT.chmod(0o555)
        print("TEST COMPLETE. Sealed receipt published. STOP for architectural review.")
        return 0
    except BaseException as exc:
        s.a.seal_json(
            s.ROOT / "failure.json",
            {
                "status": "STOPPED_FOR_ARCHITECTURAL_REVIEW",
                "phase": "test",
                "model": active_name,
                "last_completed_stage": stage,
                "exception_class": s.a.safe_exception(exc),
                "worker_exit_code": code,
                "signal": -code if code is not None and code < 0 else None,
                "raw_exception_messages_retained": False,
                "reservation_retained": True,
                "automatic_retry": False,
                "timestamp_utc": s.a.stamp(),
            },
        )
        print("TEST stopped; retained reservation/evidence requires review. No retry.")
        return 2


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    sys.addaudithook(s.a.deny_network)
    if len(sys.argv) != 1:
        return 2
    try:
        return execute()
    except BaseException:
        print("TEST preflight blocked; no TEST values loaded. Inspect safe retained evidence.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
