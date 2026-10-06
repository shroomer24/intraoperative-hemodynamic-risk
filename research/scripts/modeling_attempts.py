"""Explicit attempt states; one authorized retry for the empty MAP-only log path."""

import hashlib
import json
import os
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from modeling_process_support import atomic_json, sanitize_output

from intraop.data.modeling import read_json, sha256

MAP_KEY = "tabpfn_map__v3p5_default_seed42"


def inspect_processes():
    """Fail closed if process inspection is unavailable; never inspect environment."""
    try:
        result = subprocess.run(
            ["/bin/ps", "-axo", "pid=,ppid=,state=,comm=,args="],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Process inspection unavailable; no hosted launch permitted") from exc
    relevant = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 4)
        if len(fields) != 5:
            continue
        pid, ppid, state, executable, command = fields
        if int(pid) == os.getpid():
            continue
        # Match Python executables plus exact worker/parent argv entries. Inspection
        # shells containing these names as source text cannot become false positives.
        if not Path(executable).name.lower().startswith("python"):
            continue
        try:
            words = shlex.split(command)
        except ValueError:
            raise RuntimeError("Unparseable Python process command; review required") from None
        scripts = {Path(word).name for word in words}
        if scripts & {"modeling_tabpfn_worker.py", "modeling_development_resume.py"} or (
            "tabpfn_map" in words and "v3p5_default_seed42" in words
        ):
            relevant.append(
                {
                    "pid": int(pid),
                    "ppid": int(ppid),
                    "state": state,
                    "command": sanitize_output(command),
                }
            )
    return relevant


def require_inactive(scanner=None):
    processes = (scanner or inspect_processes)()
    if processes:
        raise RuntimeError(
            "ACTIVE hosted worker or continuation parent; no retry/launch permitted: "
            + json.dumps(processes)
        )
    return {"status": "NO_RELEVANT_ACTIVE_PROCESS", "timestamp_utc": datetime.now(UTC).isoformat()}


def scientific_identity(plan, plan_hash, model, cid):
    definition = plan["models"][model]
    candidate = next(c for c in definition["candidates"] if c["candidate_id"] == cid)
    value = {
        "model": model,
        "candidate_id": cid,
        "definition": definition,
        "candidate": candidate,
        "development_plan_sha256": plan_hash,
        "python_executable": plan["python_executable"],
        "python_version": plan["python_version"],
        "package_versions": plan["package_versions"],
        "feature_sets": plan["feature_sets"],
        "dataset_manifest_hashes": plan["dataset_manifest_hashes"],
        "training_context": plan["training_context"],
        "selection_metric": plan["selection_metric"],
        "tie_break": plan["tie_break"],
        "positive_class": 1,
        "probability_column": 1,
        "required_classes": [0, 1],
        "context_partition": "training",
        "query_partition": "tuning",
        "positive_class_semantics": "new sustained hypotension episode beginning within 5 minutes",
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def ledger_dir(output, key):
    return output / "attempt_ledgers" / key


def preserve_empty_attempt(output, plan, plan_hash, *, process_evidence):
    """Append a ledger outside attempt 1. Never write in its original directory."""
    model, cid = "tabpfn_map", "v3p5_default_seed42"
    paths = output / "process_candidates"
    logs = paths / f"{MAP_KEY}_logs"
    if not logs.is_dir() or list(logs.iterdir()):
        raise ValueError("Authorized recovery requires the exact original empty MAP log directory")
    if (paths / MAP_KEY).exists() or (output / "candidates" / f"{MAP_KEY}.csv").exists():
        raise ValueError("Completed artifact exists; attempt 1 cannot be recorded for retry")
    stat = logs.stat()
    record = {
        "candidate_id": cid,
        "model": model,
        "attempt_number": 1,
        "status": "indeterminate",
        "reason": "log directory created but no child completion evidence retained",
        "fit_outcome": "unknown",
        "hosted_request_reached_server": "unknown",
        "recorded_utc": datetime.now(UTC).isoformat(),
        "existing_evidence_paths": [str(logs.relative_to(output))],
        "original_directory_identity": {
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
            "entry_names": [],
        },
        "scientific_identity_sha256": scientific_identity(plan, plan_hash, model, cid),
        "development_plan_sha256": plan_hash,
        "process_inspection": process_evidence,
        "no_success_failure_or_server_inference": True,
    }
    directory = ledger_dir(output, MAP_KEY)
    atomic_json(directory / "attempt_1.json", record, exclusive=True)
    atomic_json(
        directory / "retry_authorization.json",
        {
            "status": "AUTHORIZED",
            "authorization_source": "explicit user authorization for MAP-only indeterminate retry",
            "model": model,
            "candidate_id": cid,
            "from_attempt": 1,
            "to_attempt": 2,
            "attempt_1_sha256": sha256(directory / "attempt_1.json"),
            "scientific_identity_sha256": record["scientific_identity_sha256"],
            "development_plan_sha256": plan_hash,
            "maximum_attempt_number": 2,
            "requires_live_process_inspection": True,
            "retry_is_additional_model_selection": False,
        },
        exclusive=True,
    )
    return record


def check_original_empty(output, first):
    logs = output / first["existing_evidence_paths"][0]
    if not logs.is_dir() or list(logs.iterdir()):
        raise ValueError("Indeterminate attempt 1 evidence changed")
    stat = logs.stat()
    identity = first["original_directory_identity"]
    for key, actual in (
        ("device", stat.st_dev),
        ("inode", stat.st_ino),
        ("mtime_ns", stat.st_mtime_ns),
        ("ctime_ns", stat.st_ctime_ns),
    ):
        if identity[key] != actual:
            raise ValueError("Indeterminate attempt 1 directory identity changed")


def attempt_state(bundle, prediction, logs, *, active=False):
    if active:
        return "ACTIVE"
    if (logs / "failure_marker.json").exists():
        return "FAILED"
    exit_path = logs / "exit.json"
    exit_record = read_json(exit_path) if exit_path.exists() else None
    if (
        exit_record
        and isinstance(exit_record.get("returncode"), int)
        and exit_record["returncode"] != 0
    ):
        return "FAILED"
    if bundle.exists():
        return "COMPLETE" if exit_record and exit_record.get("returncode") == 0 else "INDETERMINATE"
    if prediction.exists() or exit_record:
        return "INDETERMINATE"
    return "INDETERMINATE" if logs.exists() else "NEW"


def retry_map_attempt(output, plan, plan_hash, *, explicitly_authorized, scanner=None):
    if not explicitly_authorized:
        raise PermissionError("INDETERMINATE MAP attempt requires explicit retry flag")
    process_evidence = require_inactive(scanner)
    root = output / "process_candidates"
    bundle, prediction = root / MAP_KEY, output / "candidates" / f"{MAP_KEY}.csv"
    if bundle.exists() or prediction.exists():
        raise PermissionError("Completed artifact blocks duplicate retry")
    directory = ledger_dir(output, MAP_KEY)
    first = read_json(directory / "attempt_1.json")
    authorization = read_json(directory / "retry_authorization.json")
    if first.get("status") != "indeterminate" or first.get("attempt_number") != 1:
        raise PermissionError("Retry requires a recorded indeterminate first attempt")
    if authorization.get("status") != "AUTHORIZED" or authorization.get("to_attempt") != 2:
        raise PermissionError("Exact retry authorization missing")
    if sha256(directory / "attempt_1.json") != authorization["attempt_1_sha256"]:
        raise ValueError("Attempt ledger hash changed")
    fingerprint = scientific_identity(plan, plan_hash, "tabpfn_map", "v3p5_default_seed42")
    if (
        fingerprint != authorization["scientific_identity_sha256"]
        or fingerprint != first["scientific_identity_sha256"]
    ):
        raise ValueError("Retry scientific configuration changed")
    original_logs = root / f"{MAP_KEY}_logs"
    if (original_logs / "exit.json").exists():
        raise PermissionError("Exit evidence exists; no duplicate retry authorized")
    if (original_logs / "failure_marker.json").exists():
        raise PermissionError("FAILED attempt remains blocked")
    check_original_empty(output, first)
    second_logs = root / f"{MAP_KEY}_attempt_2_logs"
    if second_logs.exists() or (directory / "attempt_2.json").exists():
        raise PermissionError("Attempt 2 already reserved/started; no further retry authorized")
    record = {
        "model": "tabpfn_map",
        "candidate_id": "v3p5_default_seed42",
        "attempt_number": 2,
        "status": "new",
        "reserved_utc": datetime.now(UTC).isoformat(),
        "log_path": str(second_logs.relative_to(output)),
        "bundle_path": str(bundle.relative_to(output)),
        "scientific_identity_sha256": fingerprint,
        "development_plan_sha256": plan_hash,
        "attempt_1_sha256": sha256(directory / "attempt_1.json"),
        "authorization_sha256": sha256(directory / "retry_authorization.json"),
        "process_inspection": process_evidence,
        "retry_is_additional_model_selection": False,
    }
    atomic_json(directory / "attempt_2.json", record, exclusive=True)
    return second_logs


def paths_for_candidate(output, plan, plan_hash, model, cid, *, retry_map=False):
    """A complete bundle is skipped by caller. Only MAP attempt 1 may be retried."""
    key = f"{model}__{cid}"
    root = output / "process_candidates"
    bundle, prediction = root / key, output / "candidates" / f"{key}.csv"
    directory = ledger_dir(output, key)
    second = directory / "attempt_2.json"
    logs = root / f"{key}_logs"
    if key == MAP_KEY and second.exists():
        record = read_json(second)
        if record["scientific_identity_sha256"] != scientific_identity(plan, plan_hash, model, cid):
            raise ValueError("Retry scientific configuration changed")
        check_original_empty(output, read_json(directory / "attempt_1.json"))
        logs = output / record["log_path"]
    state = attempt_state(bundle, prediction, logs)
    if state == "COMPLETE":
        return bundle, logs, state
    require_inactive()
    if state == "FAILED":
        raise RuntimeError("FAILED candidate attempt retained; architectural review required")
    if bundle.exists() or prediction.exists() or (logs / "exit.json").exists():
        raise RuntimeError("Candidate completion evidence is incomplete; duplicate request blocked")
    if key == MAP_KEY and second.exists():
        raise RuntimeError("Attempt 2 is indeterminate/unstarted; further retry requires review")
    if state == "INDETERMINATE":
        if key != MAP_KEY:
            raise RuntimeError("INDETERMINATE candidate requires separately authorized retry")
        logs = retry_map_attempt(output, plan, plan_hash, explicitly_authorized=retry_map)
        return bundle, logs, "NEW"
    # A normal, genuinely new candidate also gets immutable attempt provenance.
    first_path = directory / "attempt_1.json"
    if first_path.exists():
        raise RuntimeError("Attempt already reserved; duplicate launch blocked")
    atomic_json(
        first_path,
        {
            "model": model,
            "candidate_id": cid,
            "attempt_number": 1,
            "status": "new",
            "reserved_utc": datetime.now(UTC).isoformat(),
            "log_path": str(logs.relative_to(output)),
            "scientific_identity_sha256": scientific_identity(plan, plan_hash, model, cid),
            "development_plan_sha256": plan_hash,
        },
        exclusive=True,
    )
    return bundle, logs, "NEW"


def record_outcome(output, model, cid, logs):
    directory = ledger_dir(output, f"{model}__{cid}")
    number = 2 if (directory / "attempt_2.json").exists() else 1
    exit_record = read_json(logs / "exit.json")
    atomic_json(
        directory / f"attempt_{number}_completion.json",
        {
            "model": model,
            "candidate_id": cid,
            "attempt_number": number,
            "status": "complete",
            "exit_record_sha256": sha256(logs / "exit.json"),
            "returncode": exit_record["returncode"],
            "bundle_manifest_sha256": sha256(
                output / "process_candidates" / f"{model}__{cid}" / "bundle_manifest.json"
            ),
            "provenance_sha256": sha256(
                output / "process_candidates" / f"{model}__{cid}" / "provenance.json"
            ),
            "published_prediction_sha256": sha256(output / "candidates" / f"{model}__{cid}.csv"),
            "timestamp_utc": datetime.now(UTC).isoformat(),
        },
        exclusive=True,
    )
