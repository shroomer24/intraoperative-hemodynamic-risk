"""Explicit, bounded acquisition from the single canonical PhysioNet release."""

import hashlib
import json
import shlex
import urllib.error
import urllib.request
from pathlib import Path

SOURCE_URL = "https://physionet.org/files/vitaldb/1.0.0/"
SOURCE_VERSION = "1.0.0"
METADATA_FILES = (
    "LICENSE.txt",
    "SHA256SUMS.txt",
    "clinical_data.csv",
    "track_names.csv",
    "clinical_parameters.csv",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_checksums(root: Path) -> dict[str, str]:
    checksums = {}
    for line in (root / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        checksums[name.lstrip("*").removeprefix("./")] = digest
    return checksums


def download_file(path: Path, url: str, *, remaining_bytes: int, on_bytes=None) -> int:
    """Stream one canonical file, counting transferred bytes including failed attempts.

    Commit only complete files; callers own selection, budgets and manifests.
    Never read more than the remaining allowance from a response.
    """
    if not url.startswith(SOURCE_URL):
        raise ValueError("Only the canonical PhysioNet release is allowed")
    if remaining_bytes <= 0:
        raise RuntimeError("Download byte budget exhausted")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    transferred = 0
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "intraop-research/0.1"})
        with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as out:
            length = (
                response.headers.get("Content-Length") if hasattr(response, "headers") else None
            )
            if length is not None and int(length) > remaining_bytes:
                raise RuntimeError("File exceeds remaining download byte budget")
            while True:
                allowance = remaining_bytes - transferred
                if allowance <= 0:
                    if length is not None and transferred == int(length):
                        break
                    raise RuntimeError("Download byte budget exhausted")
                block = response.read(min(1024 * 1024, allowance))
                if not block:
                    break
                transferred += len(block)
                if on_bytes is not None:
                    on_bytes(len(block))
                out.write(block)
            if length is not None and transferred != int(length):
                raise RuntimeError("Incomplete response")
        partial.replace(path)
    except (OSError, urllib.error.URLError, RuntimeError, ValueError):
        partial.unlink(missing_ok=True)
        raise
    return transferred


def acquire_subset(
    destination: Path, case_ids: list[int], *, max_total_bytes: int = 1024**3
) -> dict:
    """At most 20 named cases; never crawl a directory or fetch an archive.

    The byte budget includes existing selected files. Partial files are removed
    on errors. Existing files are checked against the release SHA256SUMS.
    """
    if not case_ids or len(case_ids) > 20 or len(set(case_ids)) != len(case_ids):
        raise ValueError("Supply 1–20 distinct case IDs explicitly")
    if any(type(case) is not int or not 1 <= case <= 6388 for case in case_ids):
        raise ValueError("VitalDB v1.0.0 case IDs must be integers from 1 through 6388")
    if type(max_total_bytes) is not int or max_total_bytes <= 0:
        raise ValueError("Download byte budget must be positive")
    destination.mkdir(parents=True, exist_ok=True)
    names = [*METADATA_FILES, *(f"vital_files/{case:04d}.vital" for case in case_ids)]
    consumed = 0
    records = []
    checksums = {}
    for name in names:
        path = destination / name
        url = SOURCE_URL + name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            try:
                download_file(path, url, remaining_bytes=max_total_bytes - consumed)
            except (OSError, urllib.error.URLError, RuntimeError) as error:
                command = f"curl --fail --location --output {shlex.quote(str(path))} {url}"
                raise RuntimeError(
                    f"PhysioNet acquisition failed for {name}: {error}. No alternate source used. "
                    f"Manual retry after creating the destination directory: {command}"
                ) from error
        consumed += path.stat().st_size
        if consumed > max_total_bytes:
            raise RuntimeError("Selected local files exceed the explicit byte budget")
        digest = sha256_file(path)
        if name == "SHA256SUMS.txt":
            checksums = source_checksums(destination)
        expected = checksums.get(name)
        if name != "SHA256SUMS.txt" and checksums and expected != digest:
            raise ValueError(f"Release checksum mismatch or absent checksum: {name}")
        records.append({"path": name, "url": url, "bytes": path.stat().st_size, "sha256": digest})
    for record in records:
        if record["path"] != "SHA256SUMS.txt":
            if checksums.get(record["path"]) != record["sha256"]:
                raise ValueError(f"Release checksum mismatch: {record['path']}")
    manifest_path = destination / "acquisition_manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    by_path = {entry["path"]: entry for entry in previous.get("files", [])}
    by_path.update({entry["path"]: entry for entry in records})
    manifest = {
        "source_url": SOURCE_URL,
        "source_version": SOURCE_VERSION,
        "dataset_doi": "10.13026/czw8-9p62",
        "license": "CC BY 4.0",
        "files": sorted(by_path.values(), key=lambda entry: entry["path"]),
        "selected_case_ids": sorted(set(previous.get("selected_case_ids", [])) | set(case_ids)),
        "release_checksums_verified": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
