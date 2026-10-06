"""Phase 0 only: copy sealed outputs and visualize existing local observations.

Run with the existing cohort Python environment (vitaldb==1.7.2), using -B.
No model capability, network capability, or scientific write capability exists.
The final export and private audit refuse overwrite, including empty directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.abc
import importlib.metadata
import json
import math
import os
import stat
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

sys.dont_write_bytecode = True
REPO = Path(os.environ["INTRAOP_SCIENTIFIC_REPO"]).resolve()
APP = Path(os.environ["INTRAOP_EXPORT_APP"]).resolve()
WORK = APP.parent / "private-export-work"
FINAL = APP / "public/replay-v01"
PRIVATE = WORK / "replay-private-v01"
STAGING = WORK / "replay-phase0-staging"
COHORT = REPO / "artifacts/vitaldb-cohort-v03"
MODELING = REPO / "artifacts/modeling-v01"
TEST = MODELING / "locked_execution/test"
RAW = REPO / "data/raw/vitaldb-1.0.0"
LOCK_HASH = "b2718a7c1ba448cc319c8594f93dc022480c573ef3dec3babd9b14dbdd021c0d"
RECEIPT_HASH = "7ee5661490f1d81f3c442f638d2e9845e3d58136ee5112b3a494c52f2815b375"
ORDER_HASH = "8310531e15a2c986c64ca60fee44d155fde89201f71ee2036a344f21a8e1c0eb"
MODELS = (
    "prevalence",
    "current_map",
    "logistic_map",
    "logistic_full",
    "xgboost",
    "tabpfn_map",
    "tabpfn_full",
)
LEARNED = MODELS[2:]
CHANNELS = ("map", "hr", "spo2", "etco2", "sbp", "dbp")
UNITS = {
    "map": "mmHg",
    "sbp": "mmHg",
    "dbp": "mmHg",
    "hr": "beats/min",
    "spo2": "%",
    "etco2": "mmHg",
}
TRACKS = {
    "map": "Solar8000/ART_MBP",
    "sbp": "Solar8000/ART_SBP",
    "dbp": "Solar8000/ART_DBP",
    "hr": "Solar8000/HR",
    "spo2": "Solar8000/PLETH_SPO2",
    "etco2": "Solar8000/ETCO2",
}
BLOCKED = (
    "sklearn",
    "xgboost",
    "tabpfn",
    "tabpfn_client",
    "torch",
    "intraop.models",
    "intraop.features",
    "intraop.evaluation",
    "intraop.inference",
    "intraop.cli",
    "intraop.data.pipeline",
    "intraop.data.labels",
    "intraop.data.modeling",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return (
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        + "\n"
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def invalid(_):
        raise ValueError("Non-finite JSON number rejected")

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def write_json(path, value, mode=0o600):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(canonical(value))
    path.chmod(mode)


def safe_child(root, relative):
    path = root / relative
    require(
        not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),
        "Unsafe evidence path",
    )
    return path


def inventory(root):
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*")) if p.is_file()}


class NoScientificCapability(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == p or fullname.startswith(p + ".") for p in BLOCKED):
            raise PermissionError("Model/scientific-execution import blocked")
        return None


def install_guard(write_roots=()):
    """Process-wide guard; cannot be disabled after installation."""
    roots = tuple(Path(p).resolve() for p in write_roots)
    require(
        not any(p.is_relative_to(REPO.resolve()) for p in roots),
        "Scientific repository cannot be an output",
    )
    require(
        not any(n == p or n.startswith(p + ".") for n in sys.modules for p in BLOCKED),
        "Preloaded model module",
    )
    sys.meta_path.insert(0, NoScientificCapability())

    def checked_write(value):
        if isinstance(value, (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(value)).resolve()
            if path.is_relative_to(REPO.resolve()) or not any(
                path.is_relative_to(p) for p in roots
            ):
                raise PermissionError("Write outside authorized export roots blocked")
        elif isinstance(value, int):
            raise PermissionError("Filesystem mutation by descriptor blocked")

    def audit(event, args):
        if event.startswith("socket.") and event != "socket.gethostname":
            raise PermissionError("Network capability blocked")
        if event in {
            "subprocess.Popen",
            "os.system",
            "os.posix_spawn",
            "os.fork",
            "pickle.find_class",
        }:
            raise PermissionError("External/model execution capability blocked")
        if event == "open":
            path, mode, flags = args
            if (
                isinstance(path, (str, bytes, os.PathLike))
                and Path(os.fsdecode(path)).name == ".tabpfn.env"
            ):
                raise PermissionError("Credential file access blocked")
            if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int)
                and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            ):
                checked_write(path)
        if event in {
            "os.remove",
            "os.rmdir",
            "os.mkdir",
            "os.chmod",
            "os.truncate",
            "os.utime",
            "os.chown",
        }:
            checked_write(args[0])
        if event in {"os.rename", "os.link", "os.symlink"}:
            checked_write(args[0])
            checked_write(args[1])

    sys.addaudithook(audit)


def tree_state():
    state = {}
    for p in sorted(REPO.rglob("*")):
        s = p.lstat()
        state[str(p.relative_to(REPO))] = {
            "bytes": s.st_size if p.is_file() else None,
            "mode": oct(stat.S_IMODE(s.st_mode)),
            "mtime_ns": s.st_mtime_ns,
            "symlink": p.is_symlink(),
        }
    return state


def bind_sources():
    """Verify published chains before reading/producing replay data."""
    bound = {}

    def bind(path, expected):
        require(path.is_file() and not path.is_symlink(), "Required source missing")
        actual = file_hash(path)
        require(actual == expected, "Scientific source hash mismatch; export stopped")
        bound[str(path.relative_to(REPO))] = {
            "sha256": actual,
            "bytes": path.stat().st_size,
            "mode": oct(stat.S_IMODE(path.stat().st_mode)),
        }

    bind(MODELING / "model_lock_manifest.json", LOCK_HASH)
    bind(TEST / "completion_receipt.json", RECEIPT_HASH)
    lock = read_json(MODELING / "model_lock_manifest.json")
    receipt = read_json(TEST / "completion_receipt.json")
    require(
        receipt["status"] == "COMPLETE" and receipt["execution_count"] == 1,
        "Final TEST not sealed complete",
    )
    require(
        receipt["model_lock_sha256"] == LOCK_HASH
        and receipt["query_source_order_sha256"] == ORDER_HASH,
        "Receipt binding mismatch",
    )
    for rel, h in receipt["files_sha256"].items():
        bind(safe_child(TEST, rel), h)
    for name, h in lock["dataset_manifest_hashes"].items():
        bind(safe_child(COHORT, name), h)
    tables = read_json(COHORT / "table_manifest.json")["tables"]
    require(tables == lock["table_hashes"], "Cohort table contract mismatch")
    for name, h in tables.items():
        bind(safe_child(COHORT, name), h)
    registry = read_json(TEST / "source_registry.json")
    for name in ("calibration_protocol", "test_protocol"):
        bind(MODELING / (name + ".json"), registry[name + "_sha256"])
    bind(
        MODELING / "locked_execution/calibration/calibration_parameters.json",
        receipt["calibration_parameters_sha256"],
    )
    prep_path = MODELING / "final-test-v01/execution_preparation.json"
    bind(prep_path, receipt["execution_preparation_sha256"])
    prep = read_json(prep_path)
    for rel, h in prep["bound_files_sha256"].items():
        bind(safe_child(REPO, rel), h)
    for rel, record in prep["protected_inventory"].items():
        p = safe_child(REPO, rel)
        bind(p, record["sha256"])
        require(
            p.stat().st_size == record["bytes"]
            and oct(stat.S_IMODE(p.stat().st_mode)) == record["mode"],
            "Protected evidence size/mode mismatch",
        )
    # Reader sources must be in the approved execution-source hash chain.
    for rel in (
        "src/data/vitaldb_reader.py",
        "src/data/preprocessing.py",
        "src/data/protocol.py",
        "src/data/datasets.py",
        "src/data/acquisition.py",
    ):
        require(rel in prep["bound_files_sha256"], "Unbound reader source")
    alignment = read_json(TEST / "alignment.json")
    require(alignment["query_source_order_sha256"] == ORDER_HASH, "Order binding mismatch")
    require(
        file_hash(TEST / "source_registry.json") == receipt["source_registry_sha256"],
        "Source registry mismatch",
    )
    provenance = read_json(COHORT / "provenance.json")
    for item in provenance["files"]:
        # Bind retained metadata without reading other operations' raw files.
        if "/" not in item["path"]:
            bind(safe_child(RAW, item["path"]), item["sha256"])
    return {"bound": bound, "lock": lock, "receipt": receipt, "alignment": alignment}


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def boolean(text):
    require(text in {"True", "False"}, "Invalid frozen boolean")
    return text == "True"


def optional_number(text):
    if text in {"", "nan"}:
        return None
    value = float(text)
    require(math.isfinite(value), "Invalid frozen numeric field")
    return value


def load_context(binding):
    import numpy as np
    import pandas as pd

    require(importlib.metadata.version("vitaldb") == "1.7.2", "Require vitaldb==1.7.2")
    metadata = csv_rows(COHORT / "metadata.csv")
    labels = np.array([int(x["label"]) for x in csv_rows(COHORT / "labels.csv")])
    split = read_json(COHORT / "realized_split_manifest.json")["subject_to_partition"]
    require(len(labels) == len(metadata), "Frozen metadata/label length mismatch")
    order = np.array(
        [i for i, x in enumerate(metadata) if split[x["subject_id"]] == "test"], dtype="<i8"
    )
    require(hashlib.sha256(order.tobytes()).hexdigest() == ORDER_HASH, "TEST source order mismatch")
    windows = [
        dict(metadata[i], label=int(labels[i]), query_position=q) for q, i in enumerate(order)
    ]
    ids = sorted({x["case_id"] for x in windows}, key=int)
    require(
        len(ids) == 22 and len(windows) == 3146 and sum(x["label"] for x in windows) == 116,
        "Frozen TEST population mismatch",
    )
    subjects = {x["subject_id"] for x in windows}
    require(
        len(subjects) == 22
        and not subjects & {s for s, part in split.items() if part == "training"},
        "Subject isolation mismatch",
    )
    predictions = {}
    for name in MODELS:
        with np.load(TEST / "predictions" / f"{name}.npz", allow_pickle=False) as p:
            predictions[name] = {key: p[key].copy() for key in p.files}
        p = predictions[name]
        value_key = "raw_score" if name == "current_map" else "raw_probability"
        keys = {"query_position", "true_label", value_key}
        if name in LEARNED:
            keys.add("calibrated_probability")
        require(
            set(p) == keys
            and np.array_equal(p["query_position"], np.arange(3146))
            and np.array_equal(p["true_label"], labels[order]),
            "Sealed model row alignment mismatch",
        )
        for key in keys - {"query_position", "true_label"}:
            require(p[key].shape == (3146,) and np.isfinite(p[key]).all(), "Invalid sealed output")
            if key != "raw_score":
                require(((p[key] >= 0) & (p[key] <= 1)).all(), "Invalid probability contract")
    require(
        np.all(
            predictions["prevalence"]["raw_probability"]
            == binding["lock"]["training_context"]["prevalence"]
        ),
        "Baseline mismatch",
    )
    require(
        read_json(TEST / "workers/current_map/score_checkpoint/manifest.json")["score_definition"]
        == "-map_latest",
        "Current MAP definition mismatch",
    )
    for name in MODELS:
        worker = TEST / "workers" / name
        if name == "current_map":
            values = np.load(worker / "score_checkpoint/scores.npy", allow_pickle=False)
        else:
            m = read_json(worker / "probability_checkpoint/manifest.json")
            require(
                m["classes"] == [0, 1] and m["probability_column"] == 1,
                "Positive class checkpoint mismatch",
            )
            values = np.load(
                worker / "probability_checkpoint/probabilities.npy", allow_pickle=False
            )[:, 1]
        key = "raw_score" if name == "current_map" else "raw_probability"
        require(np.array_equal(values, predictions[name][key]), "Prediction/checkpoint mismatch")
    audits = {str(x["case_id"]): x for x in read_json(COHORT / "cohort_report.json")["cases"]}
    clinical = pd.read_csv(
        RAW / "clinical_data.csv", usecols=["caseid", "subjectid", "opstart", "opend", "caseend"]
    )
    cases = []
    for index, original in enumerate(ids, 1):
        row = clinical.loc[clinical.caseid == int(original)]
        require(len(row) == 1, "Ambiguous clinical routing")
        row = row.iloc[0]
        subject_ids = {x["subject_id"] for x in windows if x["case_id"] == original}
        require(subject_ids == {str(int(row.subjectid))}, "Case/subject routing mismatch")
        path = RAW / "vital_files" / f"{int(original):04d}.vital"
        require(path.is_file() and not path.is_symlink(), "Required local raw file missing")
        h = file_hash(path)
        require(h == audits[original]["reader_audit"]["file_sha256"], "Raw source hash mismatch")
        binding["bound"][str(path.relative_to(REPO))] = {
            "sha256": h,
            "bytes": path.stat().st_size,
            "mode": oct(stat.S_IMODE(path.stat().st_mode)),
        }
        cases.append(
            {
                "ordinal": f"case-{index:03d}",
                "original": original,
                "subject": next(iter(subject_ids)),
                "clinical": row,
                "raw_path": path,
                "audit": audits[original],
            }
        )
    episodes = [x for x in csv_rows(COHORT / "episodes.csv") if x["case_id"] in ids]
    anchors = [x for x in csv_rows(COHORT / "anchor_audit.csv") if x["case_id"] in ids]
    return {
        "binding": binding,
        "cases": cases,
        "windows": windows,
        "predictions": predictions,
        "episodes": episodes,
        "anchors": anchors,
    }


def reader_modules():
    # Resolve the authoritative sources rather than an older installed scaffold.
    for name, folder in (("intraop", REPO / "src"), ("intraop.data", REPO / "src/data")):
        require(name not in sys.modules, "Unexpected preloaded scientific namespace")
        module = ModuleType(name)
        module.__path__ = [str(folder)]
        sys.modules[name] = module
    reader = importlib.import_module("intraop.data.vitaldb_reader")
    display = importlib.import_module("intraop.data.preprocessing")
    protocol = importlib.import_module("intraop.data.protocol")
    require(
        protocol.TRACKS == TRACKS
        and protocol.MAX_AGE_SECONDS == 10
        and protocol.MAP_TECHNICAL_CEILING == 250,
        "Frozen display protocol mismatch",
    )
    require(
        Path(reader.__file__).resolve() == REPO / "src/data/vitaldb_reader.py"
        and Path(display.__file__).resolve() == REPO / "src/data/preprocessing.py",
        "Wrong scientific reader module",
    )
    return reader, display


def number_list(values):
    return [float(x) if math.isfinite(float(x)) else None for x in values]


def window_record(row, offset, predictions):
    q = row["query_position"]
    anchor = float(row["anchor_time_seconds"]) - offset
    outputs = {}
    for name in MODELS:
        if name == "current_map":
            outputs[name] = {
                "raw_score": float(predictions[name]["raw_score"][q]),
                "score_definition": "-map_latest",
                "unit": "mmHg",
            }
        else:
            outputs[name] = {"raw_probability": float(predictions[name]["raw_probability"][q])}
            if name in LEARNED:
                outputs[name]["calibrated_probability"] = float(
                    predictions[name]["calibrated_probability"][q]
                )
    onset = optional_number(row["matched_episode_onset_seconds"])
    return {
        "query_position": q,
        "anchor_seconds": anchor,
        "history_start_seconds": float(row["history_start_seconds"]) - offset,
        "history_end_seconds": float(row["history_end_seconds"]) - offset,
        "horizon_start_exclusive_seconds": anchor,
        "horizon_end_inclusive_seconds": anchor + 300,
        "future_observation_end_seconds": float(row["future_observation_end_seconds"]) - offset,
        "eligible": True,
        "model_outputs": outputs,
        "ground_truth": {
            "label": row["label"],
            "matched_episode_onset_seconds": None if onset is None else onset - offset,
        },
    }


def event_records(rows, offset):
    return [
        {
            "presentation_event_ordinal": i,
            "onset_seconds": float(x["onset_time_seconds"]) - offset,
            "confirmation_seconds": float(x["confirmation_time_seconds"]) - offset,
            "is_recurrent": boolean(x["is_recurrent"]),
            "evaluable": boolean(x["evaluable"]),
            "minimum_confirmed_low_seconds": 60,
            "sustained_duration_seconds": None,
        }
        for i, x in enumerate(
            sorted(rows, key=lambda x: (float(x["onset_time_seconds"]), int(x["episode_number"]))),
            1,
        )
    ]


def anchor_records(rows, offset, windows):
    by_time = {float(x["anchor_time_seconds"]): x for x in windows}
    require(len(by_time) == len(windows), "Duplicate case forecast anchor")
    result = []
    for row in sorted(rows, key=lambda x: float(x["anchor_time_seconds"])):
        time = float(row["anchor_time_seconds"])
        require(row["status"] in {"eligible", "ineligible", "censored"}, "Unknown frozen status")
        q = None
        if row["status"] == "eligible":
            require(
                time in by_time and float(row["label"]) == by_time[time]["label"],
                "Anchor/window label join mismatch",
            )
            q = by_time[time]["query_position"]
        else:
            require(time not in by_time, "Unavailable anchor unexpectedly has forecast")
        result.append(
            {
                "anchor_seconds": time - offset,
                "frozen_status": row["status"],
                "retrospective_reason": row["reason"] or None,
                "query_position": q,
            }
        )
    require(
        sum(x["query_position"] is not None for x in result) == len(windows),
        "Eligible anchor join incomplete",
    )
    return result


def coverage(values):
    total = len(values)
    gaps = 0
    longest = run = 0
    for x in values:
        if x is None:
            if not run:
                gaps += 1
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    missing = sum(x is None for x in values)
    return {
        "grid_seconds": total,
        "valid_seconds": total - missing,
        "missing_seconds": missing,
        "valid_fraction": (total - missing) / total,
        "gap_runs": gaps,
        "longest_gap_seconds": longest,
    }


def model_definitions(lock):
    names = {
        "prevalence": "TRAINING prevalence baseline",
        "current_map": "Current MAP score",
        "logistic_map": "Logistic MAP-only",
        "logistic_full": "Logistic full",
        "xgboost": "XGBoost",
        "tabpfn_map": "TabPFN-3.5 MAP-only",
        "tabpfn_full": "TabPFN-3.5 full",
    }
    return [
        {
            "id": name,
            "display_name": names[name],
            "feature_count": len(lock["models"][name]["features"]),
            "representation": "score" if name == "current_map" else "probability",
            "available_representations": ["raw", "calibrated"] if name in LEARNED else ["raw"],
            "calibration_provenance": "frozen Platt mapping; retained calibrated TEST output"
            if name in LEARNED
            else None,
            "locked_definition_sha256": read_json(TEST / "source_registry.json")["sources"][name][
                "locked_definition_sha256"
            ],
        }
        for name in MODELS
    ]


def export_one(destination, context, reader, display):
    import validate_replay_export as validator

    require(not destination.exists(), "Export destination already exists")
    destination.mkdir(parents=True)
    entries = []
    per_case = {}
    for case in context["cases"]:
        ordinal, original = case["ordinal"], case["original"]
        print(
            json.dumps({"status": "READING_LOCAL", "pass": destination.name, "case": ordinal}),
            flush=True,
        )
        source = reader.read_local_case(RAW, case["clinical"])
        require(
            source.audit["file_sha256"] == case["audit"]["reader_audit"]["file_sha256"],
            "Reader raw source hash mismatch",
        )
        processed = display.preprocess_case(source)  # display only, never scientific tables
        offset = source.opstart_seconds
        grid = processed.grid
        signals = {
            "time_seconds": number_list(grid.time_seconds.to_numpy() - offset),
            "channels": {
                name: {
                    "unit": UNITS[name],
                    "values": number_list(grid[name]),
                    "age_seconds": number_list(grid[f"{name}_age_seconds"]),
                }
                for name in CHANNELS
            },
        }
        validator.validate_signals(signals)
        validator.validate_signal_source(signals, source)
        frozen_valid = case["audit"]["signal_audit"]["valid_seconds"]
        require(
            len(grid) == case["audit"]["signal_audit"]["surgical_grid_seconds"],
            "Frozen visualization grid length mismatch",
        )
        for name in CHANNELS:
            require(
                sum(x is not None for x in signals["channels"][name]["values"])
                == frozen_valid[name],
                "Frozen channel coverage mismatch",
            )
        windows = [x for x in context["windows"] if x["case_id"] == original]
        events = event_records([x for x in context["episodes"] if x["case_id"] == original], offset)
        statuses = anchor_records(
            [x for x in context["anchors"] if x["case_id"] == original], offset, windows
        )
        folder = destination / "cases" / ordinal
        write_json(folder / "signals.json", signals)
        write_json(
            folder / "prediction-windows.json",
            [window_record(x, offset, context["predictions"]) for x in windows],
        )
        write_json(folder / "events.json", events)
        write_json(folder / "anchor-status.json", statuses)
        case_coverage = {name: coverage(signals["channels"][name]["values"]) for name in CHANNELS}
        per_case[ordinal] = case_coverage
        entries.append(
            {
                "case_ordinal": ordinal,
                "duration_seconds": min(source.opend_seconds, source.recording_end_seconds)
                - offset,
                "grid_start_seconds": signals["time_seconds"][0],
                "grid_step_seconds": 1,
                "confirmed_episode_count": len(events),
                "evaluable_episode_count": sum(x["evaluable"] for x in events),
                "positive_window_count": sum(x["label"] for x in windows),
                "prediction_window_count": len(windows),
                "channel_coverage": case_coverage,
                "assets": {
                    name: f"cases/{ordinal}/{name}.json"
                    for name in ("signals", "prediction-windows", "anchor-status", "events")
                },
            }
        )
    write_json(destination / "cases/index.json", entries)
    receipt = context["binding"]["receipt"]
    manifest = {
        "schema_version": "replay-v01",
        "model_lock_sha256": LOCK_HASH,
        "test_completion_receipt_sha256": RECEIPT_HASH,
        "source_registry_sha256": receipt["source_registry_sha256"],
        "query_order_sha256": ORDER_HASH,
        "generator_source_sha256": file_hash(__file__),
        "time_basis": "seconds from surgical start",
        "probability_source": "sealed TEST outputs",
        "case_count": len(entries),
        "prediction_window_count": len(context["windows"]),
        "models": model_definitions(context["binding"]["lock"]),
        "license_attributions": [
            {
                "name": "VitalDB 1.0.0",
                "license": "CC BY 4.0",
                "source": "https://physionet.org/content/vitaldb/1.0.0/",
                "doi": "10.13026/czw8-9p62",
                "changes": (
                    "Held-out operation subset; relative time; "
                    "causal numeric display grid; presentation ordinals"
                ),
            }
        ],
        "presentation_files_sha256": inventory(destination),
    }
    write_json(destination / "manifest.json", manifest)
    return per_case


def verify_unchanged(bound, state):
    require(tree_state() == state, "Scientific repository file state changed")
    for rel, record in bound.items():
        p = safe_child(REPO, rel)
        require(
            file_hash(p) == record["sha256"]
            and p.stat().st_size == record["bytes"]
            and oct(stat.S_IMODE(p.stat().st_mode)) == record["mode"],
            "Bound scientific source changed",
        )


def seal_tree(root):
    for p in root.rglob("*"):
        if p.is_file():
            p.chmod(0o444)
    for p in sorted((x for x in root.rglob("*") if x.is_dir()), reverse=True):
        p.chmod(0o555)
    root.chmod(0o555)


def run():
    require(
        not FINAL.exists() and not PRIVATE.exists(),
        "Final export/audit exists; overwrite forbidden",
    )
    require(
        not PRIVATE.resolve().is_relative_to((APP / "public").resolve()), "Private audit exposed"
    )
    STAGING.mkdir(parents=True, exist_ok=True)
    os.chmod(STAGING, 0o700)
    os.environ["MPLCONFIGDIR"] = str(STAGING / "matplotlib-cache")
    install_guard((STAGING, APP, PRIVATE))
    state = tree_state()
    binding = bind_sources()
    context = load_context(binding)
    reader, display = reader_modules()
    import validate_replay_export as validator

    with tempfile.TemporaryDirectory(prefix="export-", dir=STAGING) as temp:
        temp = Path(temp)
        first, second = temp / "pass-1", temp / "pass-2"
        case_coverage = export_one(first, context, reader, display)
        summary = validator.validate_export(first, context)
        export_one(second, context, reader, display)
        validator.validate_export(second, context)
        hashes = inventory(first)
        require(hashes == inventory(second), "Deterministic exports differ")
        verify_unchanged(binding["bound"], state)
        require(
            not any(n == p or n.startswith(p + ".") for n in sys.modules for p in BLOCKED),
            "Model module loaded",
        )
        PRIVATE.mkdir(mode=0o700)
        crosswalk = [
            {
                "case_ordinal": c["ordinal"],
                "original_case_id": c["original"],
                "original_subject_id": c["subject"],
                "raw_file": str(c["raw_path"]),
                "surgical_start_recording_seconds": float(c["clinical"].opstart),
                "query_positions": [
                    x["query_position"] for x in context["windows"] if x["case_id"] == c["original"]
                ],
            }
            for c in context["cases"]
        ]
        write_json(PRIVATE / "source_crosswalk.json", crosswalk)
        write_json(
            PRIVATE / "source_hashes.json",
            {
                "scientific_root": str(REPO),
                "bound_files": binding["bound"],
                "bound_content_manifest_sha256": digest(binding["bound"]),
                "source_repo_state_sha256": digest(state),
                "source_repo_state": state,
                "generator_sha256": file_hash(__file__),
                "validator_sha256": file_hash(APP / "tools/validate_replay_export.py"),
                "schema_sha256": file_hash(APP / "tools/replay_schema.json"),
            },
        )
        audit = {
            "status": "COMPLETE",
            "scope": "Phase 0 presentation export only",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "python_executable": sys.executable,
            "python_version": sys.version,
            "versions": {n: importlib.metadata.version(n) for n in ("vitaldb", "numpy", "pandas")},
            "generator_sha256": file_hash(__file__),
            "public_manifest_sha256": hashes["manifest.json"],
            "public_files_sha256": hashes,
            "public_file_inventory": sorted(hashes),
            "two_export_hashes_identical": True,
            "determinism_passes": [digest(hashes), digest(inventory(second))],
            "source_integrity": {
                "bound_files_verified": len(binding["bound"]),
                "repo_state_unchanged": True,
                "bound_hashes_unchanged": True,
            },
            "counts": summary["counts"],
            "coverage": summary["coverage"],
            "case_coverage": case_coverage,
            "capability_guards": {
                "network_blocked": True,
                "repository_writes_blocked": True,
                "model_imports_blocked": True,
                "external_execution_blocked": True,
            },
            "scientific_calls": {
                "model_fit": 0,
                "calibration_fit": 0,
                "model_prediction": 0,
                "episode_detection": 0,
                "labeling": 0,
                "scientific_table_generation": 0,
            },
            "reader_calls": 44,
            "reader_calls_scope": "two validated determinism passes in this invocation",
            "presentation_case_processing_calls": 44,
            "independent_causal_oracle_passes": 44,
            "private_source_records_sha256": file_hash(PRIVATE / "source_hashes.json"),
            "private_crosswalk_sha256": file_hash(PRIVATE / "source_crosswalk.json"),
        }
        FINAL.parent.mkdir(parents=True, exist_ok=True)
        require(not FINAL.exists(), "Final export appeared; publication stopped")
        first.rename(FINAL)
        seal_tree(FINAL)
        verify_unchanged(binding["bound"], state)
        write_json(PRIVATE / "export_audit.json", audit)
        print(
            json.dumps(
                {
                    "status": "PHASE_0_COMPLETE",
                    "export_root": str(FINAL),
                    "public_manifest_sha256": hashes["manifest.json"],
                    "generator_sha256": file_hash(__file__),
                    "counts": summary["counts"],
                    "coverage": summary["coverage"],
                    "determinism": True,
                    "bound_source_files_verified": len(binding["bound"]),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()
