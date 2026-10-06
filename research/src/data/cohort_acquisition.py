"""Outcome-blind candidate planning and resumable task-wide download limits."""

import json
import time
from pathlib import Path

import numpy as np

from intraop.data.acquisition import SOURCE_URL, download_file, sha256_file, source_checksums
from intraop.data.vitaldb_reader import load_clinical_metadata, population_eligibility

HARD_CAP_BYTES = 8_000_000_000


def candidate_plan(clinical, pilot_cases: list[int], *, seed: int, target: int = 150) -> dict:
    """Read only population fields/arterial-line hint, never outcomes or features."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid seed")
    if type(target) is not int or not len(pilot_cases) <= target <= 150:
        raise ValueError("Target must include pilot and be at most 150 cases")
    eligible = clinical.loc[clinical.apply(population_eligibility, axis=1).isna()]
    if set(pilot_cases) - set(eligible.caseid.astype(int)):
        raise ValueError("Pilot case missing from eligible clinical candidates")
    hint = eligible.aline1.fillna("").astype(str).str.strip()
    hinted = eligible.loc[~hint.isin(["", "0", "None", "nan"])]
    ids = sorted(set(hinted.caseid.astype(int)) - set(pilot_cases))
    shuffled = np.random.default_rng(seed).permutation(ids).tolist()
    order = sorted(pilot_cases) + shuffled
    mapping = clinical.set_index("caseid").subjectid.astype(int)
    return {
        "seed": seed,
        "target_cases": target,
        "policy": "adult/general existing eligibility; nonempty aline1 acquisition hint; "
        "sorted IDs then seeded permutation; pilot prefix; no outcome fields",
        "pilot_case_ids": sorted(pilot_cases),
        "candidate_order": order,
        "candidate_subject_ids": {str(c): str(mapping.loc[c]) for c in order},
    }


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _continued_checkpoint(root, plan, parent_path, max_new_bytes, max_seconds):
    """Verify a finalized prefix before creating a separate transfer ledger."""
    parent = json.loads(parent_path.read_text())
    expected_plan = dict(parent["plan"], target_cases=plan["target_cases"])
    if parent["source_url"] != SOURCE_URL or expected_plan != plan:
        raise ValueError("Continuation source/plan differs from frozen parent")
    prefix = parent["completed_case_ids"]
    if not parent.get("frozen_for_processing") or not prefix:
        raise ValueError("Continuation requires a finalized nonempty checkpoint")
    if prefix != plan["candidate_order"][: len(prefix)] or len(prefix) > plan["target_cases"]:
        raise ValueError("Continuation parent is not a valid completed prefix")
    if [r["case_id"] for r in parent["files"]] != prefix:
        raise ValueError("Continuation parent records differ from completed prefix")
    checksums = source_checksums(root)
    for record in parent["files"]:
        name = f"vital_files/{record['case_id']:04d}.vital"
        local = root / name
        if (
            record["path"] != name
            or record["url"] != SOURCE_URL + name
            or local.stat().st_size != record["bytes"]
            or sha256_file(local) != record["sha256"]
            or checksums.get(name) != record["sha256"]
        ):
            raise ValueError(f"Continuation source identity/integrity failure: {name}")
    return {
        "plan": plan,
        "source_url": SOURCE_URL,
        "max_new_bytes": max_new_bytes,
        "max_seconds_per_invocation": max_seconds,
        "initial_local_files": sorted(p.name for p in (root / "vital_files").glob("*.vital")),
        "transferred_new_vital_bytes": 0,
        "files": [dict(r, initially_local=True) for r in parent["files"]],
        "completed_case_ids": prefix,
        "candidate_cases_considered": len(prefix),
        "stop_reason": None,
        "continuation": {
            "parent_manifest_sha256": sha256_file(parent_path),
            "parent_completed_case_ids": prefix,
            "parent_transferred_new_vital_bytes": parent["transferred_new_vital_bytes"],
            "prefix_release_checksums_verified": True,
        },
        "elapsed_acquisition_seconds": 0.0,
        "attempts": [],
    }


def acquire_cohort(
    root: Path,
    output: Path,
    pilot_cases: list[int],
    *,
    seed=42,
    target=150,
    max_new_bytes=HARD_CAP_BYTES,
    max_seconds=1800,
    resume_from: Path | None = None,
) -> dict:
    """Stop at first failure/cap/time limit; never replace cases based on labels.

    The persisted plan is frozen BEFORE opening any new recording. Resuming a
    task preserves its cumulative transfer ledger, including failed transfers.
    Only the completed deterministic prefix is returned for processing.
    """
    if type(max_new_bytes) is not int or not 0 < max_new_bytes <= HARD_CAP_BYTES:
        raise ValueError("New .vital data cap must be in (0, 8 GB]")
    if max_seconds <= 0:
        raise ValueError("Acquisition time budget must be positive")
    plan = candidate_plan(load_clinical_metadata(root), pilot_cases, seed=seed, target=target)
    plan["clinical_metadata_sha256"] = sha256_file(root / "clinical_data.csv")
    plan["release_checksums_sha256"] = sha256_file(root / "SHA256SUMS.txt")
    path = output / "acquisition_manifest.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result["plan"] != plan or result["max_new_bytes"] != max_new_bytes:
            raise ValueError("Refusing to change frozen acquisition plan/budget")
        if resume_from is not None and result.get("continuation", {}).get(
            "parent_manifest_sha256"
        ) != sha256_file(resume_from):
            raise ValueError("Continuation parent changed")
        if result.get("frozen_for_processing"):
            for record in result["files"]:
                if sha256_file(root / record["path"]) != record["sha256"]:
                    raise ValueError("Frozen source file changed")
            return result
    elif resume_from is not None:
        result = _continued_checkpoint(root, plan, resume_from, max_new_bytes, max_seconds)
        _save(path, result)
    else:
        initial = sorted(p.name for p in (root / "vital_files").glob("*.vital"))
        result = {
            "plan": plan,
            "source_url": SOURCE_URL,
            "max_new_bytes": max_new_bytes,
            "max_seconds_per_invocation": max_seconds,
            "initial_local_files": initial,
            "transferred_new_vital_bytes": 0,
            "files": [],
            "completed_case_ids": [],
            "stop_reason": None,
        }
        _save(path, result)
    checksums = source_checksums(root)
    records = {r["case_id"]: r for r in result["files"]}
    completed = []
    remaining = plan["candidate_order"][:target]
    if "continuation" in result:
        completed = list(result["completed_case_ids"])
        if (
            completed != remaining[: len(completed)]
            or [r["case_id"] for r in result["files"]] != completed
        ):
            raise ValueError("Continuation ledger is not an exact completed prefix")
        for record in result["files"]:
            local = root / record["path"]
            if (
                sha256_file(local) != record["sha256"]
                or checksums.get(record["path"]) != record["sha256"]
                or local.stat().st_size != record["bytes"]
            ):
                raise ValueError("Previously completed continuation source changed")
        remaining = remaining[len(completed) :]
    started = time.monotonic()
    prior_elapsed = result.get("elapsed_acquisition_seconds", 0.0)

    def record_elapsed():
        if "continuation" in result:
            result["elapsed_acquisition_seconds"] = prior_elapsed + time.monotonic() - started

    for case_id in remaining:
        name = f"vital_files/{case_id:04d}.vital"
        local = root / name
        if case_id not in records and time.monotonic() - started >= max_seconds:
            result["stop_reason"] = "acquisition_time_budget"
            break
        result["candidate_cases_considered"] = len(completed) + 1
        _save(path, result)
        attempt_started = time.monotonic()
        bytes_before = result["transferred_new_vital_bytes"]
        is_new_attempt = case_id not in records
        try:
            if not local.exists():

                def count_bytes(number):
                    result["transferred_new_vital_bytes"] += number
                    record_elapsed()
                    _save(path, result)

                download_file(
                    local,
                    SOURCE_URL + name,
                    remaining_bytes=max_new_bytes - result["transferred_new_vital_bytes"],
                    on_bytes=count_bytes,
                )
            digest = sha256_file(local)
            if checksums.get(name) != digest:
                raise ValueError(f"Release checksum mismatch: {name}")
        except (OSError, RuntimeError, ValueError) as error:
            result["stop_reason"] = f"{type(error).__name__}: {error}"
            if "continuation" in result:
                result["attempts"].append(
                    {
                        "case_id": case_id,
                        "subject_id": plan["candidate_subject_ids"][str(case_id)],
                        "path": name,
                        "url": SOURCE_URL + name,
                        "success": False,
                        "reason": result["stop_reason"],
                        "transferred_bytes": result["transferred_new_vital_bytes"] - bytes_before,
                        "elapsed_seconds": time.monotonic() - attempt_started,
                    }
                )
            break
        records[case_id] = {
            "case_id": case_id,
            "path": name,
            "url": SOURCE_URL + name,
            "bytes": local.stat().st_size,
            "sha256": digest,
            "initially_local": local.name in result["initial_local_files"],
            "release_checksum_verified": True,
        }
        if "continuation" in result and is_new_attempt:
            result["attempts"].append(
                {
                    **records[case_id],
                    "subject_id": plan["candidate_subject_ids"][str(case_id)],
                    "success": True,
                    "reason": None,
                    "transferred_bytes": result["transferred_new_vital_bytes"] - bytes_before,
                    "elapsed_seconds": time.monotonic() - attempt_started,
                }
            )
        completed.append(case_id)
        result["files"] = [records[c] for c in completed]
        result["completed_case_ids"] = completed
        result["stop_reason"] = "target_reached" if len(completed) == target else None
        record_elapsed()
        _save(path, result)
        print(
            f"Acquired/verified {len(completed)}/{target}; "
            f"new transfer {result['transferred_new_vital_bytes']:,} bytes",
            flush=True,
        )
    record_elapsed()
    _save(path, result)
    return result
