"""External-terminal orchestration: B then C then D, immutable A comparator."""

import fcntl
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import recovery_support as recovery
from study_support import (
    BASELINE,
    IDS,
    SPEC_HASH,
    STUDY,
    atomic_bytes,
    atomic_json,
    context_manifest,
    digest_json,
    load_development,
    matrices,
    paired_bootstrap,
    read_json,
    require_consistent_metadata,
    reserve_attempt,
    selected_representation,
    sha256,
    timestamp,
    validate_predictions,
    verified_spec,
    verify_preparation,
)


def check_bundle(directory, tuning, variant, prep):
    bundle = directory / "bundle"
    manifest = read_json(bundle / "bundle_manifest.json")
    allowed = {"prediction.csv", "metrics.json", "provenance.json", "remote_metadata.json"}
    if (
        manifest["status"] != "COMPLETE"
        or manifest["variant"] != variant
        or set(manifest["files_sha256"]) != allowed
    ):
        raise ValueError("Candidate bundle incomplete or unexpected")
    for name, digest in manifest["files_sha256"].items():
        if sha256(bundle / name) != digest:
            raise ValueError("Candidate bundle hash changed")
    reservation = read_json(recovery.attempt_path(variant) / "reservation.json")
    if reservation["scientific_identity_sha256"] != digest_json(prep["candidates"][variant]):
        raise ValueError("Scientific identity changed")
    rows, metrics = validate_predictions(bundle / "prediction.csv", tuning, variant=variant)
    stored = read_json(bundle / "metrics.json")
    if stored["variant"] != variant or stored["simplicity_rank"] != IDS.index(variant):
        raise ValueError("Metric record identity differs")
    for name, value in metrics.items():
        if isinstance(value, float):
            if abs(stored[name] - value) > 1e-12:
                raise ValueError("Candidate metrics inconsistent")
        elif stored[name] != value:
            raise ValueError("Candidate metric counts differ")
    prov = read_json(bundle / "provenance.json")
    if (
        prov["prepared_candidate_sha256"] != digest_json(prep["candidates"][variant])
        or prov["partition_access_log"] != ["training", "tuning"]
        or prov["classes"] != [0, 1]
        or prov["positive_probability_column"] != 1
        or prov["attempt_number"] != recovery.attempt_number(variant)
        or prov["execution_recovery_authorization_sha256"]
        != sha256(recovery.RECOVERY / "B_attempt_2_authorization.json")
    ):
        raise ValueError("Candidate provenance differs")
    return rows, stored, read_json(bundle / "remote_metadata.json")


def supervise(variant, directory, logs):
    logs.mkdir(parents=True, exist_ok=False)
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            str(recovery.RECOVERY / "recovery_worker.py"),
            "--variant",
            variant,
            "--attempt",
            str(recovery.attempt_number(variant)),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    # SDK output is never written: retain only the worker's narrow, validated protocol.
    protocol = {}
    try:
        parsed = json.loads(result.stdout)
        if parsed.get("variant") == variant and parsed.get("status") in {
            "COMPLETE",
            "WORKER_FAILED",
        }:
            protocol = {k: parsed[k] for k in ["status", "variant"]}
            kind = parsed.get("error_type")
            if isinstance(kind, str) and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,80}", kind):
                protocol["error_type"] = kind
    except (ValueError, TypeError, AttributeError):
        pass
    atomic_json(logs / "stdout.json", protocol, exclusive=True)
    atomic_bytes(
        logs / "stderr.txt", b"Raw child stderr is deliberately not persisted.\n", exclusive=True
    )
    exit_record = {
        "variant": variant,
        "returncode": result.returncode,
        "signal": -result.returncode if result.returncode < 0 else None,
        "worker_protocol": protocol,
        "raw_streams_discarded": True,
        "timestamp_utc": timestamp(),
        **recovery.stage_summary(recovery.attempt_path(variant)),
    }
    exit_record["hosted_outcome"] = (
        "COMPLETE"
        if result.returncode == 0 and protocol.get("status") == "COMPLETE"
        else "INDETERMINATE"
        if exit_record["hosted_call_begun"] or exit_record["http_request_started"]
        else "NO_HOSTED_CALL_START_RECORDED"
    )
    failure = recovery.attempt_path(variant) / "failure.json"
    if failure.exists():
        observed = read_json(failure)
        status = observed.get("http_failure_status")
        if recovery.http_category(status) is not None:
            exit_record["http_failure_status"] = status
            exit_record["http_failure_category"] = recovery.http_category(status)
    atomic_json(logs / "exit.json", exit_record, exclusive=True)
    if result.returncode != 0:
        atomic_json(
            logs / "failure_marker.json", {"status": "FAILED", **exit_record}, exclusive=True
        )
        raise RuntimeError("Candidate worker failed; evidence retained, no automatic retry")
    if protocol.get("status") != "COMPLETE":
        raise RuntimeError("Successful exit has unconfirmed completion; no duplicate request")


def complete_attempt(directory, variant):
    attempt = recovery.attempt_path(variant)
    receipt = attempt / "completion.json"
    value = {
        "status": "complete",
        "variant": variant,
        "attempt_number": recovery.attempt_number(variant),
        "bundle_manifest_sha256": sha256(directory / "bundle/bundle_manifest.json"),
        "exit_record_sha256": sha256(attempt / "logs/exit.json"),
        "scientific_identity_sha256": read_json(attempt / "reservation.json")[
            "scientific_identity_sha256"
        ],
        "independent_candidate_evaluation_number": 1,
    }
    if receipt.exists():
        if read_json(receipt) != value:
            raise ValueError("Completed attempt receipt changed")
    else:
        atomic_json(receipt, value, exclusive=True)


def publish_once(path, payload):
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("Immutable published result differs")
    else:
        atomic_bytes(path, payload, exclusive=True)


def review_zip():
    target = STUDY / "tabpfn_representation_study_review.zip"
    manifest = read_json(STUDY / "study_completion_manifest.json")
    if target.exists():
        with zipfile.ZipFile(target) as z:
            if z.testzip() is not None:
                raise ValueError("Existing review ZIP is damaged")
            for name, digest in {
                **manifest["files_sha256"],
                "study_completion_manifest.json": sha256(STUDY / "study_completion_manifest.json"),
                "study_completion_manifest.sha256": sha256(
                    STUDY / "study_completion_manifest.sha256"
                ),
            }.items():
                import hashlib

                if (
                    hashlib.sha256(z.read("tabpfn-representation-study-v01/" + name)).hexdigest()
                    != digest
                ):
                    raise ValueError("Existing review ZIP file hash differs")
        return target
    fd, temp = tempfile.mkstemp(prefix=".review-", suffix=".zip", dir=STUDY)
    os.close(fd)
    temporary = Path(temp)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as z:
            for name in [
                *manifest["files_sha256"],
                "study_completion_manifest.json",
                "study_completion_manifest.sha256",
            ]:
                z.write(STUDY / name, "tabpfn-representation-study-v01/" + name)
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def verify_complete():
    manifest = read_json(STUDY / "study_completion_manifest.json")
    if (
        sha256(STUDY / "study_completion_manifest.json")
        != (STUDY / "study_completion_manifest.sha256").read_text().strip()
    ):
        raise ValueError("Completion manifest hash changed")
    if manifest["status"] != "COMPLETE" or manifest["study_plan_sha256"] != SPEC_HASH:
        raise ValueError("Completion manifest differs")
    for name, digest in manifest["files_sha256"].items():
        if sha256(STUDY / name) != digest:
            raise ValueError("Completed study artifact changed")
    return manifest


def finish(plan, prep, training, tuning, records, probabilities, reports):
    consistency = require_consistent_metadata(reports)
    intervals = paired_bootstrap(tuning, probabilities)
    a = records[0]["average_precision"]
    rows = [
        {
            **r,
            "AP_difference_vs_A": r["average_precision"] - a,
            "AP_difference_CI_low": 0.0
            if r["variant"] == IDS[0]
            else intervals["AP_difference_95pct_percentile"][r["variant"]][0],
            "AP_difference_CI_high": 0.0
            if r["variant"] == IDS[0]
            else intervals["AP_difference_95pct_percentile"][r["variant"]][1],
        }
        for r in records
    ]
    root = STUDY / "execution"
    publish_once(root / "comparison.csv", pd.DataFrame(rows).to_csv(index=False).encode())
    publish_once(root / "bootstrap.json", (json.dumps(intervals, indent=2) + "\n").encode())
    winner = selected_representation(records)
    choice = {
        "selected_variant": winner["variant"],
        "tuning_metrics": winner,
        "definition": next(v for v in plan["variants"] if v["id"] == winner["variant"]),
        "criterion": "highest TUNING AP; frozen tie-breaks and 1e-12 tolerance",
        "bootstrap_used_for_selection": False,
        "selection_is_final_for_this_study": True,
        "final_model_lock_written": False,
    }
    publish_once(
        root / "selected_representation.json", (json.dumps(choice, indent=2) + "\n").encode()
    )
    publish_once(
        root / "remote_consistency.json", (json.dumps(consistency, indent=2) + "\n").encode()
    )
    publish_once(root / "retained_baseline_A_prediction.csv", BASELINE.read_bytes())
    d = context_manifest(training, matrices(training, tuning, plan, IDS[3])[3])
    if d != read_json(STUDY / "D_context_selection_manifest.json"):
        raise ValueError("D context evidence changed")
    # Recheck frozen inputs, baseline and preparation source integrity after all workers.
    verified_spec()
    recovery.verify_authorization(plan, prep, training, tuning)
    verify_preparation(plan)
    load_development(plan)
    inventory = {
        str(p.relative_to(STUDY)): sha256(p)
        for p in sorted(STUDY.rglob("*"))
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.suffix in {".py", ".sh", ".md", ".json", ".csv", ".sha256", ".txt"}
        and p.name not in {"study_completion_manifest.json"}
    }
    manifest = {
        "status": "COMPLETE",
        "completed_utc": timestamp(),
        "study_plan_sha256": SPEC_HASH,
        "preparation_manifest_sha256": sha256(STUDY / "preparation_manifest.json"),
        "variant_order": list(IDS),
        "new_scientific_candidates_evaluated": 3,
        "published_successful_fit_calls": 3,
        "published_successful_predict_proba_calls": 3,
        "attempt_1_fit_predict_calls": "UNKNOWN",
        "B_attempt_1_operational_state": "FAILED",
        "B_attempt_1_hosted_outcome": "INDETERMINATE",
        "B_successful_attempt": 2,
        "B_execution_recovery_authorization_sha256": sha256(
            recovery.RECOVERY / "B_attempt_2_authorization.json"
        ),
        "baseline_A_rerun": False,
        "selected_representation": winner["variant"],
        "comparison": rows,
        "D_context": d,
        "remote_runtime_consistency": consistency,
        "validation": read_json(recovery.RECOVERY / "validation_report.json"),
        "original_study_validation": read_json(STUDY / "validation_report_v02.json"),
        "files_sha256": inventory,
        "partition_access_log": ["training", "tuning"],
        "calibration_values_parsed": False,
        "test_values_parsed": False,
        "model_lock_written": False,
        "credential_values_recorded": False,
        "no_follow_up_optimization": True,
        "review_zip": "tabpfn_representation_study_review.zip",
    }
    atomic_json(STUDY / "study_completion_manifest.json", manifest, exclusive=True)
    atomic_bytes(
        STUDY / "study_completion_manifest.sha256",
        (sha256(STUDY / "study_completion_manifest.json") + "\n").encode(),
        exclusive=True,
    )
    review_zip()
    return manifest


def run():
    plan, prep, loader, training, tuning, auth, env = recovery.checked_context(external=True)
    if (recovery.RECOVERY / "study_stop.json").exists():
        raise RuntimeError("New unresolved failure blocks the entire study")
    if (STUDY / "study_completion_manifest.json").exists():
        result = verify_complete()
        review_zip()
        return result
    recovery.require_inactive()
    root = STUDY / "execution"
    existed = root.exists()
    root.mkdir(exist_ok=True)
    with (root / ".execution.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("ACTIVE orchestration lock; no duplicate execution") from None
        # checked_context cryptographically verified this exact historical stop.
        # Its original file and all B1 FAILED evidence are retained unchanged.
        if (
            sha256(root / "review_required.json")
            != auth["historical_execution_records_sha256"]["review_required.json"]
        ):
            raise RuntimeError("Unrecognized historical stop remains blocking")
        start = root / "started_manifest.json"
        if existed and not start.exists():
            raise RuntimeError("INDETERMINATE execution directory retained; review required")
        identity = {
            "study_plan_sha256": SPEC_HASH,
            "preparation_manifest_sha256": sha256(STUDY / "preparation_manifest.json"),
            "baseline_A_prediction_sha256": sha256(BASELINE),
        }
        if start.exists():
            if read_json(start)["identity"] != identity:
                raise ValueError("Started study identity changed")
        else:
            atomic_json(
                start,
                {"status": "STARTED", "identity": identity, "timestamp_utc": timestamp()},
                exclusive=True,
            )
        _, a_metrics = validate_predictions(BASELINE, tuning)
        baseline = pd.read_csv(BASELINE).predicted_probability.to_numpy()
        records = [{"variant": IDS[0], "simplicity_rank": 0, **a_metrics}]
        probabilities = {IDS[0]: baseline}
        reports = []
        try:
            for variant in IDS[1:]:
                directory = root / "candidates" / variant
                state = recovery.state(variant)
                if state == "AUTHORIZED_NOT_ALLOCATED":
                    logs = recovery.reserve_B_attempt_2(plan, prep, training, tuning)
                    print(
                        json.dumps({"status": "STARTING", "variant": variant, "attempt": 2}),
                        flush=True,
                    )
                    supervise(variant, directory, logs)
                elif state == "NEW":
                    print(json.dumps({"status": "STARTING", "variant": variant}), flush=True)
                    logs = reserve_attempt(
                        directory,
                        digest_json(prep["candidates"][variant]),
                        scanner=recovery.inspect_processes,
                    )
                    supervise(variant, directory, logs)
                elif state != "COMPLETE":
                    raise RuntimeError(
                        f"Candidate state {state}; review required, no automatic retry"
                    )
                frame, record, remote = check_bundle(directory, tuning, variant, prep)
                reports.append(remote)
                require_consistent_metadata(reports)  # stop before next variant or comparison
                complete_attempt(directory, variant)
                records.append(record)
                probabilities[variant] = frame.predicted_probability.to_numpy()
                print(
                    json.dumps(
                        {
                            "status": "VERIFIED_COMPLETE",
                            "variant": variant,
                            "reused_existing_result": state == "COMPLETE",
                        }
                    ),
                    flush=True,
                )
            if loader.access_log != ["training", "tuning"]:
                raise PermissionError("Partition access differs")
            return finish(plan, prep, training, tuning, records, probabilities, reports)
        except Exception as exc:
            if not (recovery.RECOVERY / "study_stop.json").exists():
                atomic_json(
                    recovery.RECOVERY / "study_stop.json",
                    {
                        "status": "STOPPED_FOR_REVIEW",
                        "error_type": recovery.safe_error_class(exc),
                        "timestamp_utc": timestamp(),
                        "raw_error_or_patient_payload_retained": False,
                        "automatic_retry_permitted": False,
                    },
                    exclusive=True,
                )
            raise


def main():
    try:
        result = run()
    except Exception:
        print(
            "Study stopped; retained attempt evidence requires review. No automatic retry.",
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "comparison": result["comparison"],
                "selected_representation": result["selected_representation"],
                "D_context": result["D_context"],
                "review_zip": str(STUDY / "tabpfn_representation_study_review.zip"),
                "model_lock_written": False,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
