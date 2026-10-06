"""Resume the authoritative partial run; fresh exec workers, no repeat smoke."""

import argparse
import fcntl
import json
import os
import sys
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from modeling_attempts import paths_for_candidate, record_outcome, require_inactive
from modeling_process_support import (
    PLAN,
    REMEDIATION,
    REPO,
    atomic_bytes,
    atomic_json,
    candidate_definition,
    original_runner,
    supervise,
    validate_prediction,
    verified_plan,
)

from intraop.data.modeling import read_json, sha256
from intraop.evaluation.development import select_candidate
from intraop.evaluation.secure_errors import sanitized_traceback


def preserve_check(output, manifest):
    """Completed outputs immutable; aggregate extensions retain original evidence."""
    baseline = REPO / manifest["preservation_directory"]
    originals = manifest["partial_files_sha256"]
    for name, digest in originals.items():
        if sha256(baseline / name) != digest:
            raise ValueError("Preserved original checkpoint hash mismatch")
        if name not in ("candidate_metrics.csv", "selected_configs.json"):
            if sha256(output / name) != digest:
                raise ValueError(f"Completed checkpoint changed: {name}")
    if (
        not (output / "candidate_metrics.csv")
        .read_bytes()
        .startswith((baseline / "candidate_metrics.csv").read_bytes())
    ):
        raise ValueError("Original candidate metric bytes changed")
    initial = read_json(baseline / "selected_configs.json")
    selected = read_json(output / "selected_configs.json")
    if any(selected.get(name) != config for name, config in initial.items()):
        raise ValueError("Previously selected conventional configurations changed")
    smoke = read_json(output / "external_training_smoke.json")
    if (
        smoke.get("status") != "PASS"
        or smoke.get("classes") != [0, 1]
        or smoke.get("positive_probability_column") != 1
    ):
        raise ValueError("Successful external smoke evidence failed integrity check")
    started = read_json(output / "started_manifest.json")
    if started["development_plan_sha256"] != manifest["original_plan_sha256"]:
        raise ValueError("Original run plan differs")
    return baseline


def candidate_keys(plan):
    return [
        (name, c["candidate_id"])
        for name, definition in plan["models"].items()
        for c in definition["candidates"]
    ]


def remaining_candidates(plan, completed):
    expected = candidate_keys(plan)
    if not completed <= set(expected):
        raise ValueError("Unexpected completed candidate")
    return [key for key in expected if key not in completed]


def bundle_check(bundle, output, tuning, model, candidate_id, plan):
    inventory = read_json(bundle / "bundle_manifest.json")
    definition, candidate = candidate_definition(plan, model, candidate_id)
    allowed = {"prediction.csv", "record.json", "provenance.json"}
    if definition["kind"] == "xgboost":
        allowed.add("model.joblib")
    if inventory["status"] != "COMPLETE" or set(inventory["files_sha256"]) != allowed:
        raise ValueError("Incomplete candidate bundle")
    for name, digest in inventory["files_sha256"].items():
        if sha256(bundle / name) != digest:
            raise ValueError("Candidate bundle hash mismatch")
    rows = validate_prediction(bundle / "prediction.csv", tuning, model, candidate_id)
    record = read_json(bundle / "record.json")
    if record["model"] != model or record["candidate_id"] != candidate_id:
        raise ValueError("Candidate record identity differs")
    if (
        json.loads(record["hyperparameters"]) != candidate["hyperparameters"]
        or record["simplicity_rank"] != candidate["simplicity_rank"]
    ):
        raise ValueError("Candidate configuration differs")
    provenance = read_json(bundle / "provenance.json")
    if provenance["training_rows"] != 13975 or provenance["query_rows"] != 3193:
        raise ValueError("Full frozen development context differs")
    if (
        provenance["partition_access_log"] != ["training", "tuning"]
        or provenance["classes"] != [0, 1]
        or provenance["positive_probability_column"] != 1
    ):
        raise ValueError("Candidate class/firewall evidence differs")
    if definition["kind"] == "xgboost" and not provenance["forbidden_runtime_modules_absent"]:
        raise ValueError("Candidate isolation evidence missing")
    target = output / "candidates" / f"{model}__{candidate_id}.csv"
    # Recover interrupted publication without refitting or replacing existing bytes.
    if target.exists():
        if sha256(target) != inventory["files_sha256"]["prediction.csv"]:
            raise ValueError("Candidate output already exists with different bytes")
    else:
        atomic_bytes(target, (bundle / "prediction.csv").read_bytes(), exclusive=True)
    return record, rows, provenance


def records_from_baseline(baseline):
    rows = pd.read_csv(baseline / "candidate_metrics.csv")
    return rows.astype(object).where(pd.notna(rows), None).to_dict("records")


def update_aggregates(output, baseline, receipts):
    # Keep the original CSV as an exact prefix, including its original float bytes.
    content = (baseline / "candidate_metrics.csv").read_bytes()
    columns = pd.read_csv(baseline / "candidate_metrics.csv", nrows=0).columns.tolist()
    if receipts:
        content += pd.DataFrame(receipts)[columns].to_csv(index=False, header=False).encode()
    atomic_bytes(output / "candidate_metrics.csv", content)


def select_and_save(output, baseline, plan, training, records, bundles):
    selected = read_json(baseline / "selected_configs.json")
    for model, definition in plan["models"].items():
        if model in selected:
            continue
        local = [r for r in records if r["model"] == model]
        if len(local) != len(definition["candidates"]):
            continue
        winner = select_candidate(local)
        cid = winner["candidate_id"]
        config = candidate_definition(plan, model, cid)[1]
        selected[model] = {
            "kind": definition["kind"],
            "features": definition["features"],
            "preprocessing": definition["preprocessing"],
            "selected_configuration": config,
            "tuning_metrics": winner,
            "selection_metric": plan["selection_metric"],
            "tie_break": plan["tie_break"],
            "fit_partition": "training",
            "selection_partition": "tuning",
            "training_rows": len(training.X),
            "training_subjects": int(training.groups.nunique()),
            "training_prevalence": float(training.y.mean()),
            "positive_class": 1,
            "classes": [0, 1],
            "probability_column": 1,
        }
        bundle = bundles[(model, cid)]
        if definition["kind"] == "xgboost":
            target = output / "models" / f"{model}.joblib"
            payload = (bundle / "model.joblib").read_bytes()
            if target.exists():
                if sha256(target) != sha256(bundle / "model.joblib"):
                    raise ValueError("Selected model checkpoint differs")
            else:
                atomic_bytes(target, payload, exclusive=True)
        else:
            provenance = {
                "model_identifier": "v3.5_default",
                "tabpfn_client_version": "0.6.1",
                "constructor_arguments": config["hyperparameters"],
                "classes": [0, 1],
                "fitted_train_rows": len(training.X),
                "hosted_inference": True,
                "training_labels_from": "training_only",
                "input_columns": definition["features"],
                "preprocessing": definition["preprocessing"],
                "remote_internals_serialized": False,
                "positive_probability_column": 1,
            }
            target = output / "models" / f"{model}.json"
            if target.exists():
                if read_json(target) != provenance:
                    raise ValueError("TabPFN provenance checkpoint differs")
            else:
                atomic_json(target, provenance, exclusive=True)
    atomic_json(output / "selected_configs.json", selected)
    return selected


def finish(output, plan, manifest, selected, records, loader, environment):
    pd_records = pd.DataFrame(
        [select_candidate([r for r in records if r["model"] == name]) for name in plan["models"]]
    )
    atomic_bytes(
        output / "tuning_metrics.csv", pd_records.to_csv(index=False).encode(), exclusive=True
    )
    predictions = [
        pd.read_csv(
            output
            / "candidates"
            / f"{model}__{selected[model]['selected_configuration']['candidate_id']}.csv",
            dtype={"window_id": str, "subject_id": str, "case_id": str},
        )
        for model in plan["models"]
    ]
    atomic_bytes(
        output / "tuning_predictions.csv",
        pd.concat(predictions, ignore_index=True).to_csv(index=False).encode(),
        exclusive=True,
    )
    preserve_check(output, manifest)
    original_runner().validate_inputs(REPO, plan)
    started = read_json(output / "started_manifest.json")
    completion = {
        "status": "COMPLETE",
        "phase": "development_only",
        "run_id": started["run_id"],
        "started_utc": started["timestamp_utc"],
        "completed_utc": datetime.now(UTC).isoformat(),
        "development_plan_sha256": manifest["original_plan_sha256"],
        "environment": environment,
        "dataset_manifest_hashes": plan["dataset_manifest_hashes"],
        "split_manifest_hash": plan["dataset_manifest_hashes"]["realized_split_manifest.json"],
        "feature_sets": plan["feature_sets"],
        "code_hashes": plan["code_hashes"],
        "process_isolation_manifest_sha256": sha256(REMEDIATION),
        "attempt_recovery_manifest_sha256": manifest.get("attempt_recovery_manifest_sha256"),
        "models_completed": list(selected),
        "candidate_count": len(records),
        "selected_configurations_file": "selected_configs.json",
        "selection_metric": plan["selection_metric"],
        "tie_break": plan["tie_break"],
        "training_context": plan["training_context"],
        "smoke_test": read_json(output / "external_training_smoke.json"),
        "smoke_rerun": False,
        "preserved_completed_candidate_count": 10,
        "partition_access_log": loader.access_log,
        "calibration_accessed": False,
        "test_values_parsed": False,
        "test_evaluation_count": 0,
        "model_lock_written": False,
        "prediction_rows_per_selected_model": 3193,
        "credential_values_recorded": False,
        "review_package": "development_review.zip",
        "files_sha256": {
            str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()
        },
    }
    # Completion is committed only after the review archive has been fully prepared.
    completion_bytes = (json.dumps(completion, indent=2, allow_nan=False) + "\n").encode()
    allowlist = [
        "started_manifest.json",
        "external_training_smoke.json",
        "candidate_metrics.csv",
        "tuning_metrics.csv",
        "selected_configs.json",
        "models/tabpfn_map.json",
        "models/tabpfn_full.json",
    ]
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        archive_path = Path(temporary) / "development_review.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in allowlist:
                archive.write(output / name, name)
            archive.writestr("development_completion_manifest.json", completion_bytes)
            archive.write(PLAN, "development_plan.json")
            archive.write(REMEDIATION, "process_isolation_manifest.json")
            recovery = REPO / "artifacts/modeling-v01/attempt_recovery_manifest.json"
            if recovery.exists():
                archive.write(recovery, "attempt_recovery_manifest.json")
                for ledger in sorted((output / "attempt_ledgers").rglob("*.json")):
                    archive.write(ledger, str(ledger.relative_to(output)))
            archive.writestr("development_plan.sha256", manifest["original_plan_sha256"] + "\n")
        atomic_bytes(output / "development_review.zip", archive_path.read_bytes(), exclusive=True)
    atomic_bytes(output / "development_completion_manifest.json", completion_bytes, exclusive=True)
    print("Development COMPLETE. STOP for architectural review.", flush=True)
    print(f"Review artifact: {output / 'development_review.zip'}", flush=True)
    return completion


def resume(conventional_only=False, validate_only=False, retry_indeterminate_tabpfn_map=False):
    plan, manifest = verified_plan()
    output = REPO / "artifacts/modeling-v01/development"
    baseline = preserve_check(output, manifest)
    if (output / "development_completion_manifest.json").exists():
        raise PermissionError("Development already complete; no further fits permitted")
    if validate_only:
        print("Resume integrity PASS; 10 original candidates and hosted smoke preserved.")
        return
    if not conventional_only:
        require_inactive()
    gates = REPO / manifest["validation_directory"]
    for mode in ("synthetic", "training"):
        evidence = read_json(gates / mode / "provenance.json")
        if evidence.get("status") != "PASS" or not evidence.get("forbidden_runtime_modules_absent"):
            raise ValueError("Required isolated validation gate failed")
        if read_json(gates / f"{mode}_logs" / "exit.json")["returncode"] != 0:
            raise ValueError("Required isolated validation child failed")
    environment = original_runner().validate_environment(plan, require_token=not conventional_only)
    loader, training, tuning = original_runner().validate_inputs(REPO, plan)
    records = records_from_baseline(baseline)
    initial_keys = {(r["model"], r["candidate_id"]) for r in records}
    if len(records) != 10 or set(remaining_candidates(plan, initial_keys)) != {
        (name, c["candidate_id"])
        for name in ("xgboost", "tabpfn_map", "tabpfn_full")
        for c in plan["models"][name]["candidates"]
    }:
        raise ValueError("Original partial candidate set differs")
    for model, cid in initial_keys:
        validate_prediction(output / "candidates" / f"{model}__{cid}.csv", tuning, model, cid)
    bundles, receipts = {}, []
    process_dir = output / "process_candidates"
    process_dir.mkdir(exist_ok=True)
    for model, cid in remaining_candidates(plan, initial_keys):
        key = f"{model}__{cid}"
        bundle = process_dir / key
        if conventional_only and plan["models"][model]["kind"] == "tabpfn" and not bundle.exists():
            continue
        logs = process_dir / f"{key}_logs"
        kind = plan["models"][model]["kind"]
        attempt = None
        if kind == "tabpfn":
            bundle, logs, attempt = paths_for_candidate(
                output,
                plan,
                manifest["original_plan_sha256"],
                model,
                cid,
                retry_map=retry_indeterminate_tabpfn_map,
            )
        if bundle.exists():
            if not logs.exists() or read_json(logs / "exit.json")["returncode"] != 0:
                raise ValueError(
                    "Existing bundle lacks successful parent supervision; review required"
                )
            print(f"Verified completed candidate; skipping fit: {model}/{cid}", flush=True)
        else:
            if (output / "candidates" / f"{key}.csv").exists():
                raise ValueError("Candidate prediction exists without a committed worker bundle")
            if logs.exists():
                raise RuntimeError(
                    "Prior candidate failure retained; architectural review required"
                )
            print(f"Fresh-process development candidate: {model}/{cid}", flush=True)
            kind = plan["models"][model]["kind"]
            worker = REPO / "scripts" / f"modeling_{kind}_worker.py"
            command = [
                plan["python_executable"],
                "-u",
                str(worker),
                "--model",
                model,
                "--candidate",
                cid,
                "--output",
                str(bundle),
            ]
            env = os.environ.copy()
            if kind == "xgboost":
                # No auth needed by the native worker; runtime/thread variables stay unchanged.
                env.pop("TABPFN_TOKEN", None)
            supervise(command, logs, env=env)
        record, _, _ = bundle_check(bundle, output, tuning, model, cid, plan)
        if kind == "tabpfn" and attempt == "NEW":
            record_outcome(output, model, cid, logs)
        records.append(record)
        receipts.append(record)
        bundles[(model, cid)] = bundle
        update_aggregates(output, baseline, receipts)
        select_and_save(output, baseline, plan, training, records, bundles)
        preserve_check(output, manifest)
    selected = select_and_save(output, baseline, plan, training, records, bundles)
    if len(records) != 20:
        print(
            "Conventional development COMPLETE: 8 XGBoost candidates. "
            "Hosted TabPFN pending normal terminal.",
            flush=True,
        )
        return
    finish(output, plan, manifest, selected, records, loader, environment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conventional-only", action="store_true", help="Never imports/runs hosted TabPFN"
    )
    parser.add_argument(
        "--retry-indeterminate-tabpfn-map",
        action="store_true",
        help="Explicitly invoke only the recorded MAP candidate attempt 2; no generic retries",
    )
    parser.add_argument("--validate-only", action="store_true", help="Checkpoint verification only")
    args = parser.parse_args()
    # Separate lock file; original directory-creation guard is never bypassed in its runner.
    guard = REPO / "artifacts/modeling-v01/process_isolation_execution.lock"
    with guard.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            resume(args.conventional_only, args.validate_only, args.retry_indeterminate_tabpfn_map)
        except Exception as exc:
            trace = sanitized_traceback(exc)
            failure = REPO / "artifacts/modeling-v01/validation/process_isolation/failures"
            failure.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            atomic_json(
                failure / f"{stamp}.json",
                {
                    "status": "FAILED",
                    "exception_type": type(exc).__name__,
                    "sanitized_traceback": trace,
                    "test_evaluation_count": 0,
                    "model_lock_written": False,
                },
                exclusive=True,
            )
            print(trace, file=sys.stderr, flush=True)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
