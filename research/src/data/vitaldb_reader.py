"""Local-only original numeric records from the pinned official VitalDB reader."""

import gzip
import struct
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from intraop.data.acquisition import sha256_file
from intraop.data.protocol import TRACKS

READER_VERSION = "1.7.2"


def recording_clock_envelope(path: Path) -> dict:
    """Recover numeric recording zero independently of selected tracks.

    New headers carry the origin explicitly. Legacy release v3 headers do not:
    scan ALL record timestamp fields for their minimum without decoding signals.
    The offset is never converted into a calendar datetime. Truncation raises.
    """
    with gzip.open(path, "rb") as handle:
        if handle.read(4) != b"VITA":
            raise ValueError("Invalid vital file signature")
        prefix = handle.read(6)
        if len(prefix) != 6:
            raise ValueError("Truncated vital header")
        length = struct.unpack_from("<H", prefix, 4)[0]
        header = handle.read(length)
        if len(header) != length:
            raise ValueError("Truncated vital header")
        explicit = struct.unpack_from("<d", header, 10)[0] if length >= 26 else None
        earliest = float("inf")
        latest = float("-inf")
        while packet_header := handle.read(5):
            if len(packet_header) != 5:
                raise ValueError("Truncated vital packet header")
            kind, size = struct.unpack("<BI", packet_header)
            payload = handle.read(size)
            if len(payload) != size:
                raise ValueError("Truncated vital packet body")
            if kind == 1:
                if len(payload) < 12:
                    raise ValueError("Truncated vital record timestamp")
                timestamp = struct.unpack_from("<d", payload, 2)[0]
                earliest = min(earliest, timestamp)
                latest = max(latest, timestamp)
        origin = explicit if explicit is not None else earliest
        if not np.isfinite(origin):
            raise ValueError("Recording clock origin unavailable")
        return {
            "source_clock_origin": float(origin),
            "origin_method": "explicit_header" if explicit is not None else "all_record_minimum",
            "first_packet_relative_seconds": earliest - origin,
            "last_packet_relative_seconds": latest - origin,
        }


def recording_clock_origin(path: Path) -> float:
    """Retained frozen origin conversion; envelope diagnostics are audit-only."""
    return recording_clock_envelope(path)["source_clock_origin"]


@dataclass(frozen=True)
class CaseSignals:
    case_id: str
    subject_id: str
    opstart_seconds: float
    opend_seconds: float
    samples: dict[str, pd.DataFrame]
    audit: dict
    recording_end_seconds: float | None = None


def population_eligibility(row: pd.Series) -> str | None:
    """Adults/general anesthesia in the canonical noncardiac release."""
    age_text = str(row.age)
    if age_text == ">89":
        age_lower_bound = 89.0
    else:
        age_lower_bound = pd.to_numeric(age_text, errors="coerce")
    if not np.isfinite(age_lower_bound) or age_lower_bound < 18:
        return "age_missing_or_under_18"
    if row.ane_type != "General":
        return "not_general_anesthesia"
    bounds = [row.opstart, row.opend, row.caseend]
    if (
        not np.isfinite(bounds).all()
        or row.opstart >= row.opend
        or max(0, row.opstart) >= min(row.opend, row.caseend)
    ):
        return "invalid_surgical_interval"
    if row.casestart != 0:
        return "unexpected_nonzero_recording_origin"
    return None


def load_clinical_metadata(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(root / "clinical_data.csv")
    required = {
        "caseid",
        "subjectid",
        "age",
        "ane_type",
        "opstart",
        "opend",
        "casestart",
        "caseend",
    }
    if required - set(frame.columns) or frame.caseid.duplicated().any():
        raise ValueError("Invalid canonical clinical_data.csv schema/case mapping")
    if frame[["caseid", "subjectid"]].isna().any().any():
        raise ValueError("Case and subject mappings cannot be missing")
    for name in ("caseid", "subjectid"):
        if ((frame[name] % 1 != 0) | (frame[name] < 1)).any():
            raise ValueError("Case/subject identifiers must be positive integers")
    return frame


def read_local_case(root: Path, row: pd.Series) -> CaseSignals:
    """Extract rec['dt']/rec['val']; never call get_samples/to_numpy/to_pandas.

    Subtract the unfiltered file recording origin from binary anonymized clock
    values. Never subtract the selected VitalFile.dtstart, which can be the first
    selected signal sample. All returned times are numeric relative seconds.
    """
    if metadata.version("vitaldb") != READER_VERSION:
        raise RuntimeError(f"Install the tested local reader: vitaldb=={READER_VERSION}")
    from vitaldb import VitalFile

    case_id = str(int(row.caseid))
    path = root / "vital_files" / f"{int(case_id):04d}.vital"
    if not path.is_file():
        raise FileNotFoundError(f"Local case missing: {path}. Run the explicit acquire command.")
    envelope = recording_clock_envelope(path)
    origin = envelope["source_clock_origin"]
    vital = VitalFile(str(path.resolve()), track_names=list(TRACKS.values()))
    samples = {}
    audits = {}
    for signal, name in TRACKS.items():
        track = vital.trks.get(name)
        if track is not None and track.type != 2:
            raise ValueError(f"Expected numeric track, found type {track.type}: {name}")
        records = [] if track is None else track.recs
        data = pd.DataFrame(
            [(float(record["dt"]) - origin, float(record["val"])) for record in records],
            columns=["time_seconds", "value"],
            dtype=float,
        )
        if not np.isfinite(data.time_seconds).all() or (data.time_seconds < 0).any():
            raise ValueError("Unexpected timestamp values in the frozen relative-second release")
        if len(data) and data.time_seconds.max() > float(row.caseend) + 60:
            raise ValueError("Record timestamps disagree with clinical relative recording bounds")
        audits[signal] = {
            "track_present": track is not None,
            "original_records": len(data),
            "duplicate_timestamps": int(data.time_seconds.duplicated().sum()),
            "out_of_order_records": int((data.time_seconds.diff() < 0).sum()),
            "nonfinite_values": int((~np.isfinite(data.value)).sum()),
        }
        samples[signal] = data.sort_values("time_seconds", kind="stable").reset_index(drop=True)
    return CaseSignals(
        case_id,
        str(int(row.subjectid)),
        float(row.opstart),
        float(row.opend),
        samples,
        {
            "file_sha256": sha256_file(path),
            "file_bytes": path.stat().st_size,
            "reader_version": READER_VERSION,
            **envelope,
            "time_conversion": "original_record_dt_minus_unfiltered_recording_origin",
            "tracks": audits,
        },
        recording_end_seconds=float(row.caseend),
    )
