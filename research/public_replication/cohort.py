"""Outcome-blind public selection using the unchanged original candidate planner."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from intraop.data.acquisition import (
    METADATA_FILES,
    SOURCE_URL,
    download_file,
    sha256_file,
    source_checksums,
)
from intraop.data.cohort_acquisition import candidate_plan
from intraop.data.vitaldb_reader import load_clinical_metadata

SCOPE = "PUBLIC REPLICATION COHORT — NOT SEALED HACKATHON RESULTS"
CASE_COUNT = 12
SEED = 42
MAX_BYTES = 2_000_000_000
HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent


def canonical_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def acquire_metadata(root: Path) -> None:
    """Use the original bounded release downloader; never discover local datasets."""
    root.mkdir(parents=True, exist_ok=True)
    consumed = 0
    for name in METADATA_FILES:
        path = root / name
        if path.is_symlink():
            raise PermissionError("Symlinked public input forbidden")
        if not path.exists():
            download_file(path, SOURCE_URL + name, remaining_bytes=MAX_BYTES - consumed)
        consumed += path.stat().st_size
    sums = source_checksums(root)
    for name in METADATA_FILES:
        if name != "SHA256SUMS.txt" and sha256_file(root / name) != sums.get(name):
            raise ValueError("Canonical public metadata checksum mismatch")


def acquire_recordings(root: Path, manifest: dict) -> None:
    """Four fixed-selection transfers using the original bounded downloader.

    Completion order never selects/replaces a case. Original acquire_subset
    subsequently verifies every canonical checksum and the complete 2 GB input cap.
    """
    names = [
        f"vital_files/{case:04d}.vital"
        for case in manifest["public_vitaldb_case_ids_in_acquisition_order"]
    ]
    spent = sum(
        (root / name).stat().st_size for name in [*METADATA_FILES, *names] if (root / name).exists()
    )
    lock = threading.Lock()
    checksums = source_checksums(root)

    def count(number):
        nonlocal spent
        with lock:
            spent += number
            if spent > MAX_BYTES:
                raise RuntimeError("Public input transfer budget exhausted")

    def acquire(name):
        path = root / name
        if path.is_symlink():
            raise PermissionError("Symlinked recording forbidden")
        if not path.exists():
            download_file(path, SOURCE_URL + name, remaining_bytes=MAX_BYTES, on_bytes=count)
        if sha256_file(path) != checksums.get(name):
            raise ValueError("Public recording release checksum mismatch")

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(acquire, names))


def derive_manifest(root: Path) -> dict:
    """No recording, outcome, pilot reservation or private roster is consulted."""
    plan = candidate_plan(load_clinical_metadata(root), [], seed=SEED, target=CASE_COUNT)
    cases = plan["candidate_order"][:CASE_COUNT]
    if len(cases) != CASE_COUNT:
        raise ValueError("Insufficient public metadata candidates; no alternative selection")
    return {
        "scope": SCOPE,
        "manifest_version": 1,
        "source_url": SOURCE_URL,
        "source_version": "1.0.0",
        "dataset_doi": "10.13026/czw8-9p62",
        "data_license": "CC BY 4.0",
        "selection": {
            "function": "intraop.data.cohort_acquisition.candidate_plan",
            "population": "original population_eligibility plus original nonempty aline1 hint",
            "ordering": "sorted integer public case IDs; numpy.default_rng(42).permutation",
            "pilot_prefix": [],
            "seed": SEED,
            "requested_cases": CASE_COUNT,
            "outcome_blind": True,
            "replacement_on_failure_or_no_anchors": False,
        },
        "public_vitaldb_case_ids_in_acquisition_order": cases,
        "public_metadata_sha256": {name: sha256_file(root / name) for name in METADATA_FILES},
        "original_selected_model_policy": "fixed recorded configurations; no new selection",
        "split_policy": "original 60/15/10/15 largest-remainder subject splitter; seed 42; "
        "no episode stratification or pilot pinning; eligible-window subjects only",
    }


def validate_manifest(root: Path) -> dict:
    actual = derive_manifest(root)
    expected = json.loads((HERE / "public_cohort_manifest.json").read_text())
    if actual != expected:
        raise ValueError("Public selection changed; do not substitute another cohort")
    return actual
