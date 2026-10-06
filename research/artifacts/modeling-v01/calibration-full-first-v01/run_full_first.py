"""First full calibration supervisor. One child, append-only evidence, mandatory review stop."""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import full_recovery_support as support  # noqa: E402


def validate_completion(output, amendment_hash):
    receipt, _ = support.verify_seal(output / "completion_receipt.json")
    if (
        receipt["status"] != "COMPLETE"
        or receipt["phase"] != "calibration"
        or receipt["model"] != "tabpfn_full"
        or receipt["attempt"] != 1
        or receipt["execution_amendment_sha256"] != amendment_hash
        or receipt["model_lock_sha256"] != support.LOCK_HASH
        or receipt["calibration_protocol_sha256"] != support.PROTOCOL_HASH
        or receipt["partition_access"] != ["training", "calibration"]
        or receipt["hosted_fit_calls"] != 1
        or receipt["predict_proba_calls"] != 1
        or receipt["automatic_retry"]
        or receipt["review_boundary"] != "STOP_BEFORE_PLATT_OR_TEST"
        or (output / "failure.json").exists()
    ):
        raise PermissionError("Worker receipt contract differs")
    for relative, expected in receipt["files_sha256"].items():
        path = output / relative
        if (
            not path.resolve().is_relative_to(output.resolve())
            or support.file_hash(path) != expected
        ):
            raise PermissionError("Worker receipt evidence differs")
    checkpoint = output / "probability_checkpoint"
    manifest, _ = support.verify_seal(checkpoint / "manifest.json")
    contract = manifest["contract"]
    if (
        manifest["status"] != "SCIENTIFICALLY_VALIDATED"
        or manifest["scientific_validation"] != "PASS"
        or manifest["identity_compatibility"]["execution_amendment_sha256"] != amendment_hash
        or contract["candidate"] != "tabpfn_full"
        or contract["constructor"] != support.CONSTRUCTOR
        or contract["query_rows"] != 2393
        or contract["query_source_order_sha256"] != support.QUERY_ORDER_HASH
        or contract["training_context_sha256"] != support.CONTEXT_HASH
        or contract["model_definition_sha256"] != support.MODEL_HASH
        or contract["feature_count"] != 74
        or support.digest(contract["features"]) != support.FEATURE_HASH
        or contract["positive_class"] != 1
        or contract["probability_column"] != 1
        or support.file_hash(checkpoint / "probabilities.npy") != manifest["probabilities_sha256"]
        or any(p.stat().st_mode & 0o222 for p in checkpoint.iterdir())
        or checkpoint.stat().st_mode & 0o222
    ):
        raise PermissionError("Immutable checkpoint contract differs")
    return receipt


def execute():
    _, amendment_hash, _ = support.checked_state(external=True)
    process_check = support.inspect_processes()
    support.reserve_attempt(amendment_hash, process_check)
    output = support.resolve_attempt()
    print(
        json.dumps(
            {"status": "STARTING", "phase": "calibration", "model": "tabpfn_full", "attempt": 1}
        ),
        flush=True,
    )
    code = None
    try:
        child = subprocess.run(
            [support.PYTHON, "-I", "-u", str(support.BASE / "full_worker.py")],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        code = child.returncode
        support.atomic_json(
            output / "process_exit.json",
            {
                "exit_code": code,
                "signal": -code if code < 0 else None,
                "raw_stdout_stderr_retained": False,
            },
        )
        if code != 0:
            raise RuntimeError("Child did not complete")
        validate_completion(output, amendment_hash)
        support.checked_state(external=True)
        support.seal_json(
            output / "supervisor_receipt.json",
            {
                "status": "COMPLETE_FULL_ONLY_STOP_FOR_REVIEW",
                "model": "tabpfn_full",
                "attempt": 1,
                "timestamp_utc": support.stamp(),
                "execution_amendment_sha256": amendment_hash,
                "worker_completion_receipt_sha256": support.file_hash(
                    output / "completion_receipt.json"
                ),
                "preserved_workers_reverified": list(support.PRESERVED),
                "tabpfn_full_first_worker_complete": True,
                "platt_started": False,
                "TEST_accessed": False,
                "automatic_retry": False,
                "second_full_attempt_authorized": False,
            },
        )
        print("First full worker complete. STOP for architectural review before Platt or TEST.")
        return 0
    except BaseException as exc:
        state = "FAILED" if (output / "failure.json").exists() else "INDETERMINATE"
        support.seal_json(
            output / "supervisor_stop.json",
            {
                "status": state,
                "review_required": True,
                "automatic_retry": False,
                "exception_class": support.safe_exception(exc),
                "exit_code": code,
                "signal": -code if code is not None and code < 0 else None,
                "timestamp_utc": support.stamp(),
                "probability_checkpoint_exists": (
                    output / "probability_checkpoint/manifest.json"
                ).exists(),
                "hosted_outcome_inferred": False,
            },
        )
        print("Attempt evidence retained. STOP for architectural review. No retry.")
        return 2


def main():
    if len(sys.argv) != 1:
        print("First-full capability accepts no arguments.")
        return 2
    try:
        return execute()
    except BaseException:
        print("Recovery guard blocked before launch; no hosted request started.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
