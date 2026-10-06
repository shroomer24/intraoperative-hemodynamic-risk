"""One local aggregate execution; verify staged artifacts before final receipt."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregation_support as s  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from intraop.evaluation.calibration import apply_platt, reliability  # noqa: E402
from intraop.evaluation.development import development_metrics  # noqa: E402
from intraop.evaluation.locked_execution import verify_calibration_complete  # noqa: E402


def validate_publication(root, registry, lock):
    files = s.receipt_files(root)
    if set(files) != s.FINAL_FILES:
        raise ValueError("Exact final artifact set required")
    s.verify_seal(root / "source_registry.json", s.REGISTRY_HASH)
    parameters, _ = s.verify_seal(root / "calibration_parameters.json")
    alignment, _ = s.verify_seal(root / "calibration_alignment.json")
    summaries, _ = s.verify_seal(root / "metrics.json")
    tables, _ = s.verify_seal(root / "reliability.json")
    if set(parameters) != set(s.CALIBRATED_MODELS):
        raise ValueError("Exactly five prespecified Platt mappings required")
    y = np.load(root / "calibration_labels.npy", allow_pickle=False)
    if (
        y.shape != (s.ROWS,)
        or y.dtype.kind not in "iu"
        or np.unique(y).tolist() != [0, 1]
        or int(y.sum()) != s.POSITIVES
        or s.file_hash(root / "calibration_labels.npy") != alignment["label_file_sha256"]
        or s.array_hash(y) != alignment["labels_array_sha256"]
    ):
        raise ValueError("Frozen calibration label binding differs")
    if (
        alignment["partition"] != "calibration"
        or alignment["rows"] != s.ROWS
        or alignment["subjects"] != s.SUBJECTS
        or alignment["positive_labels"] != s.POSITIVES
        or alignment["query_source_order_sha256"] != s.QUERY_ORDER_HASH
        or alignment["source_registry_sha256"] != s.REGISTRY_HASH
        or alignment["model_lock_sha256"] != s.LOCK_HASH
        or alignment["calibration_protocol_sha256"] != s.PROTOCOL_HASH
        or alignment["label_table_sha256"] != lock["table_hashes"]["labels.csv"]
        or alignment["metadata_routing_table_sha256"] != lock["table_hashes"]["metadata.csv"]
        or alignment["split_manifest_sha256"] != lock["split_manifest_hash"]
        or alignment["ordinal_position_sha256"] != s.row_hash(np.arange(s.ROWS))
        or alignment["raw_model_configuration_digest"] != s.digest(lock["models"])
        or alignment["patient_identifiers_published"]
        or not alignment["calibration_metrics_diagnostic_only"]
    ):
        raise ValueError("Label/source/ordinal alignment contract differs")
    raw = s.raw_outputs(registry)
    expected_metrics, expected_tables = {}, {}
    for name in s.MODEL_NAMES:
        if name in parameters:
            mapping = parameters[name]
            if (
                mapping["method"] != "Platt_logit_logistic"
                or mapping["epsilon"] != 1e-6
                or mapping["fit_partition"] != "calibration"
                or mapping["configuration_changed"]
                or mapping["fitting_parameters"] != s.CALIBRATION_PROTOCOL["parameters"]
                or not np.isfinite([mapping["coefficient"], mapping["intercept"]]).all()
            ):
                raise ValueError("Platt method/parameters differ")
            calibrated = apply_platt(raw[name], mapping)
        else:
            calibrated = None
        with np.load(root / "predictions" / f"{name}.npz", allow_pickle=False) as arrays:
            expected = {
                "query_position",
                "true_label",
                "raw_score" if name == "current_map" else "raw_probability",
            }
            if calibrated is not None:
                expected.add("calibrated_probability")
            if set(arrays.files) != expected:
                raise ValueError("Prediction keys must exclude identifiers and extra transforms")
            for key, value in {
                "query_position": np.arange(s.ROWS),
                "true_label": y,
                "raw_score" if name == "current_map" else "raw_probability": raw[name],
                **({"calibrated_probability": calibrated} if calibrated is not None else {}),
            }.items():
                if not np.array_equal(arrays[key], value):
                    raise ValueError("Published predictions or alignment differ")
        frame = pd.read_csv(root / "predictions" / f"{name}.csv", float_precision="round_trip")
        if frame.columns.tolist() != [
            "query_position",
            "true_label",
            "raw_probability",
            "raw_score",
            "calibrated_probability",
            "model",
            "split",
        ]:
            raise ValueError("Prediction CSV columns differ or contain identifiers")
        if (
            len(frame) != s.ROWS
            or not np.array_equal(frame.query_position, np.arange(s.ROWS))
            or not np.array_equal(frame.true_label, y)
            or not frame.model.eq(name).all()
            or not frame.split.eq("calibration").all()
        ):
            raise ValueError("Prediction CSV alignment differs")
        probability = name != "current_map"
        column = "raw_probability" if probability else "raw_score"
        unused = "raw_score" if probability else "raw_probability"
        if not np.array_equal(frame[column], raw[name]) or not frame[unused].isna().all():
            raise ValueError("Raw predictions or score/probability discipline differ")
        if calibrated is None:
            if not frame.calibrated_probability.isna().all():
                raise ValueError("Fixed prevalence/current MAP cannot be recalibrated")
        elif not np.array_equal(frame.calibrated_probability, calibrated):
            raise ValueError("Calibrated CSV probabilities differ")
        metric = development_metrics(y, raw[name], probability_output=probability)
        if not probability:
            metric = {
                k: metric[k]
                for k in [
                    "windows",
                    "positives",
                    "average_precision",
                    "auroc",
                    "probability_output",
                ]
            }
        expected_metrics[name + "/raw"] = metric
        if probability:
            expected_tables[name + "/raw"] = reliability(y, raw[name])
        if calibrated is not None:
            expected_metrics[name + "/calibrated"] = development_metrics(
                y, calibrated, probability_output=True
            )
            expected_tables[name + "/calibrated"] = reliability(y, calibrated)
    if summaries != expected_metrics or tables != expected_tables:
        raise ValueError("Fixed diagnostic/reliability artifacts differ")
    if (root / "reliability.png").read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Required reliability figure missing or invalid")
    if any(p.stat().st_mode & 0o222 for p in root.rglob("*") if p.is_file()):
        raise PermissionError("Final artifacts must be read-only")
    return alignment, parameters, files


def execute():
    ph = s.reserve()
    code = None
    try:
        print("STARTING one local prespecified calibration aggregation.", flush=True)
        child = subprocess.run(
            [s.PYTHON, "-I", "-u", str(s.BASE / "aggregation_worker.py")],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        code = child.returncode
        s.atomic_json(
            s.ATTEMPT / "process_exit.json",
            {
                "exit_code": code,
                "signal": -code if code < 0 else None,
                "raw_stdout_stderr_retained": False,
            },
        )
        if code != 0:
            raise RuntimeError("Local worker did not complete")
        prep, current, lock, registry = s.checked_state(external=True)
        if current != ph:
            raise PermissionError("Execution preparation changed")
        worker, _ = s.verify_seal(s.ATTEMPT / "worker_completion.json")
        if (
            worker["status"] != "STAGED_COMPLETE"
            or worker["execution_preparation_sha256"] != ph
            or worker["source_registry_sha256"] != s.REGISTRY_HASH
            or worker["partition_access"] != ["calibration"]
            or worker["Platt_models"] != list(s.CALIBRATED_MODELS)
            or worker["Platt_fit_calls"] != 5
            or worker["raw_model_fit_or_predict_calls"] != 0
            or worker["hosted_calls"] != 0
            or worker["TEST_accessed"]
            or worker["automatic_retry"]
        ):
            raise ValueError("Worker completion contract differs")
        publication = s.ATTEMPT / "publication"
        if s.receipt_files(publication) != worker["files_sha256"]:
            raise PermissionError("Staged artifact hashes differ")
        alignment, parameters, files = validate_publication(publication, registry, lock)
        s.checked_state(external=True)
        for relative in sorted(s.FINAL_FILES):
            s.copy_immutable(publication / relative, s.CALIBRATION / relative)
        final_files = {
            relative: s.file_hash(s.CALIBRATION / relative) for relative in s.FINAL_FILES
        }
        if final_files != files:
            raise PermissionError("Published artifact hashes differ")
        s.checked_state(external=True)
        s.seal_json(
            s.RECEIPT,
            {
                "status": "COMPLETE",
                "phase": "calibration",
                "models": list(s.MODEL_NAMES),
                "model_lock_sha256": s.LOCK_HASH,
                "calibration_protocol_sha256": s.PROTOCOL_HASH,
                "rows": s.ROWS,
                "subjects": s.SUBJECTS,
                "query_source_order_sha256": s.QUERY_ORDER_HASH,
                "source_registry_sha256": s.REGISTRY_HASH,
                "execution_preparation_sha256": ph,
                "execution_source_sha256": prep["bound_files_sha256"],
                "MAP_recovery_amendment_sha256": s.MAP_AMENDMENT_HASH,
                "full_model_amendment_sha256": s.FULL_AMENDMENT_HASH,
                "label_alignment_sha256": s.file_hash(s.CALIBRATION / "calibration_alignment.json"),
                "Platt_models": list(s.CALIBRATED_MODELS),
                "Platt_parameter_set_sha256": {
                    name: s.digest(parameters[name]) for name in s.CALIBRATED_MODELS
                },
                "final_artifacts_sha256": final_files,
                "files_sha256": s.receipt_files(s.CALIBRATION),
                "raw_model_fit_or_predict_calls": 0,
                "hosted_calls": 0,
                "TEST_accessed": False,
                "automatic_rerun": False,
                "calibration_configuration_changes": False,
                "calibration_metrics_are_in_sample_secondary": True,
                "diagnostic_metrics_cannot_affect_TEST_configuration": True,
                "preserved_historical_failures_not_overwritten_or_reclassified": True,
                "timestamp_utc": s.stamp(),
            },
        )
        verify_calibration_complete(s.MODELING, s.LOCK_HASH)
        for p in s.ATTEMPT.rglob("*"):
            p.chmod(0o555 if p.is_dir() else 0o444)
        s.ATTEMPT.chmod(0o555)
        (s.CALIBRATION / "predictions").chmod(0o555)
        print("CALIBRATION complete. STOP for architectural review. TEST remains unavailable.")
        return 0
    except BaseException as exc:
        if s.ATTEMPT.stat().st_mode & 0o222:
            s.seal_json(
                s.ATTEMPT / "supervisor_stop.json",
                {
                    "status": "FAILED"
                    if (s.ATTEMPT / "failure.json").exists() or code is not None and code >= 0
                    else "INDETERMINATE",
                    "exception_class": s.safe_exception(exc),
                    "exit_code": code,
                    "signal": -code if code is not None and code < 0 else None,
                    "review_required": True,
                    "automatic_retry": False,
                    "raw_exception_messages_retained": False,
                    "timestamp_utc": s.stamp(),
                },
            )
        print("Aggregation stopped; evidence retained. No retry. Architectural review required.")
        return 2


def main():
    s.install_execution_firewall()
    if len(sys.argv) != 1:
        print("Calibration aggregation accepts no arguments.")
        return 2
    try:
        return execute()
    except BaseException:
        print("Aggregation guard blocked before launch. No fitting performed.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
