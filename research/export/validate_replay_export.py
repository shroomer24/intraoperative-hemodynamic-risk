"""Offline schema, frozen-output, case-routing and source-integrity validation."""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path

import export_replay as e


def validate_schema(value, schema, document):
    """Validate the closed Draft 2020-12 subset used by our local schema.

    No remote references, code generation or third-party schema runtime.
    This validates our declared subset, not arbitrary JSON Schema documents.
    """
    if "$ref" in schema:
        ref = schema["$ref"]
        e.require(ref.startswith("#/$defs/"), "External schema reference forbidden")
        return validate_schema(value, document["$defs"][ref.split("/")[-1]], document)
    if "anyOf" in schema:
        for variant in schema["anyOf"]:
            try:
                validate_schema(value, variant, document)
                break
            except ValueError:
                continue
        else:
            raise ValueError("Schema alternatives mismatch")
    if "type" in schema:
        kind = schema["type"]
        types = kind if isinstance(kind, list) else [kind]
        passed = any(
            {
                "null": value is None,
                "boolean": type(value) is bool,
                "integer": type(value) is int,
                "number": type(value) in (int, float) and math.isfinite(value),
                "string": isinstance(value, str),
                "array": isinstance(value, list),
                "object": isinstance(value, dict),
            }[k]
            for k in types
        )
        e.require(passed, "Schema type mismatch")
    if "const" in schema:
        e.require(
            type(value) is type(schema["const"]) and value == schema["const"],
            "Schema constant mismatch",
        )
    if "enum" in schema:
        e.require(value in schema["enum"], "Schema enumeration mismatch")
    if type(value) in (int, float):
        e.require(math.isfinite(value), "Non-finite number")
        if "minimum" in schema:
            e.require(value >= schema["minimum"], "Schema lower bound")
        if "maximum" in schema:
            e.require(value <= schema["maximum"], "Schema upper bound")
    if isinstance(value, str) and "pattern" in schema:
        e.require(re.fullmatch(schema["pattern"], value) is not None, "Schema pattern mismatch")
    if isinstance(value, list):
        if "minItems" in schema:
            e.require(len(value) >= schema["minItems"], "Schema minimum items")
        if "maxItems" in schema:
            e.require(len(value) <= schema["maxItems"], "Schema maximum items")
        for item in value:
            validate_schema(item, schema.get("items", {}), document)
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        e.require(set(schema.get("required", [])) <= set(value), "Missing schema field")
        for key, child in value.items():
            if key in properties:
                validate_schema(child, properties[key], document)
            else:
                additional = schema.get("additionalProperties", True)
                e.require(additional is not False, "Unapproved public field")
                if isinstance(additional, dict):
                    validate_schema(child, additional, document)


def privacy_check(value):
    banned = {
        "subject_id",
        "case_id",
        "caseid",
        "subjectid",
        "original_case_id",
        "source_case_id",
        "raw_file",
        "source_clock_origin",
        "demographics",
    }
    if isinstance(value, dict):
        e.require(not set(value) & banned, "Public identifier/clock field detected")
        for child in value.values():
            privacy_check(child)
    elif isinstance(value, list):
        for child in value:
            privacy_check(child)
    elif isinstance(value, str):
        e.require(
            ".vital" not in value.lower() and "/Users/" not in value, "Public source path detected"
        )
        e.require(
            not re.search(r"\b\d{4}-\d{2}-\d{2}(?:T|\b)", value),
            "Public calendar timestamp detected",
        )


def validate_signals(signals):
    times = signals["time_seconds"]
    e.require(
        bool(times) and all(type(t) in (float, int) and math.isfinite(t) and t >= 0 for t in times),
        "Invalid signal clock",
    )
    e.require(all(b - a == 1 for a, b in zip(times, times[1:], strict=False)), "Non-unit grid")
    e.require(set(signals["channels"]) == set(e.CHANNELS), "Channel set mismatch")
    for name, channel in signals["channels"].items():
        values, ages = channel["values"], channel["age_seconds"]
        e.require(
            channel["unit"] == e.UNITS[name] and len(values) == len(ages) == len(times),
            "Signal alignment/unit mismatch",
        )
        for value, age in zip(values, ages, strict=True):
            e.require(age is None or math.isfinite(age) and age >= 0, "Invalid observation age")
            if value is not None:
                e.require(
                    math.isfinite(value) and age is not None and age <= 10, "Invalid fresh value"
                )
                if name == "map":
                    e.require(0 < value <= 250, "Invalid MAP display sample")
            else:
                e.require(age is None or age > 10, "Fresh value incorrectly missing")


def validate_signal_source(signals, source):
    """Independent forward-scan oracle over original local reader observations.

    Uses neither preprocess_case nor causal_hold to compute expected displays.
    Fixtures in tests are numeric algorithm fixtures, never exported medical data.
    """
    end = min(source.opend_seconds, source.recording_end_seconds)
    grid = range(max(0, math.ceil(source.opstart_seconds)), math.floor(end) + 1)
    expected_times = [float(t - source.opstart_seconds) for t in grid]
    e.require(signals["time_seconds"] == expected_times, "Source grid/time-offset mismatch")
    for name in e.CHANNELS:
        valid = [
            (float(t), float(v))
            for t, v in source.samples[name].itertuples(index=False, name=None)
            if t <= end and math.isfinite(v) and (name != "map" or 0 < v <= 250)
        ]
        e.require(
            all(a[0] <= b[0] for a, b in zip(valid, valid[1:], strict=False)),
            "Unordered source samples",
        )
        j = 0
        last = None
        channel = signals["channels"][name]
        for i, t in enumerate(grid):
            while j < len(valid) and valid[j][0] <= t:
                last = valid[j]
                j += 1
            age = None if last is None else t - last[0]
            value = None if last is None or age > 10 else last[1]
            e.require(
                channel["values"][i] == value and channel["age_seconds"][i] == age,
                "Source causal hold/freshness/duplicate/gap mismatch",
            )


def validate_export(root, context=None):
    root = Path(root)
    schema = e.read_json(e.APP / "tools/replay_schema.json")
    manifest = e.read_json(root / "manifest.json")
    hashes = e.inventory(root)
    e.require(
        len(hashes) == 90
        and hashes.keys() == {"manifest.json", *manifest["presentation_files_sha256"]},
        "Public file inventory mismatch",
    )
    e.require(
        {k: v for k, v in hashes.items() if k != "manifest.json"}
        == manifest["presentation_files_sha256"],
        "Public asset hash mismatch",
    )
    e.require(
        manifest["generator_source_sha256"] == e.file_hash(e.__file__)
        and manifest["model_lock_sha256"] == e.LOCK_HASH
        and manifest["test_completion_receipt_sha256"] == e.RECEIPT_HASH
        and manifest["query_order_sha256"] == e.ORDER_HASH,
        "Public provenance mismatch",
    )
    for rel in hashes:
        e.require(not (root / rel).is_symlink(), "Public symlink forbidden")
        value = e.read_json(root / rel)
        e.require((root / rel).read_bytes() == e.canonical(value), "Noncanonical public JSON")
        privacy_check(value)
        kind = {"manifest.json": "manifest", "cases/index.json": "case_index"}.get(rel)
        if kind is None:
            kind = Path(rel).stem.replace("-", "_")
        validate_schema(value, {"$ref": "#/$defs/" + kind}, schema)
    entries = e.read_json(root / "cases/index.json")
    e.require(
        [x["case_ordinal"] for x in entries] == [f"case-{i:03d}" for i in range(1, 23)],
        "Stable case ordinal mismatch",
    )
    rows = []
    episodes = positives = evaluable = 0
    status_counts = Counter()
    channel_totals = {name: Counter() for name in e.CHANNELS}
    for entry in entries:
        ordinal = entry["case_ordinal"]
        paths = {
            name: f"cases/{ordinal}/{name}.json"
            for name in ("signals", "prediction-windows", "anchor-status", "events")
        }
        e.require(entry["assets"] == paths, "Cross-case asset routing mismatch")
        signals, windows, statuses, events = (
            e.read_json(root / paths[x])
            for x in ("signals", "prediction-windows", "anchor-status", "events")
        )
        validate_signals(signals)
        e.require(
            entry["grid_start_seconds"] == signals["time_seconds"][0]
            and entry["duration_seconds"] >= signals["time_seconds"][-1],
            "Case bounds mismatch",
        )
        e.require(
            all(
                b["anchor_seconds"] > a["anchor_seconds"]
                for a, b in zip(windows, windows[1:], strict=False)
            ),
            "Case window order mismatch",
        )
        e.require(
            all(
                b["anchor_seconds"] > a["anchor_seconds"]
                for a, b in zip(statuses, statuses[1:], strict=False)
            ),
            "Case scheduled anchor order mismatch",
        )
        for name in e.CHANNELS:
            cov = e.coverage(signals["channels"][name]["values"])
            e.require(entry["channel_coverage"][name] == cov, "Coverage summary mismatch")
            for key in ("grid_seconds", "valid_seconds", "missing_seconds", "gap_runs"):
                channel_totals[name][key] += cov[key]
            channel_totals[name]["longest_gap_seconds"] = max(
                channel_totals[name]["longest_gap_seconds"], cov["longest_gap_seconds"]
            )
        e.require(
            len(events) == entry["confirmed_episode_count"]
            and sum(x["evaluable"] for x in events) == entry["evaluable_episode_count"],
            "Event count mismatch",
        )
        e.require(
            [x["presentation_event_ordinal"] for x in events] == list(range(1, len(events) + 1)),
            "Event ordinal mismatch",
        )
        e.require(
            all(
                x["confirmation_seconds"] >= x["onset_seconds"]
                and x["sustained_duration_seconds"] is None
                for x in events
            ),
            "Invented duration/order",
        )
        e.require(
            len(windows) == entry["prediction_window_count"]
            and sum(x["ground_truth"]["label"] for x in windows) == entry["positive_window_count"],
            "Case forecast count mismatch",
        )
        by_anchor = {x["anchor_seconds"]: x["query_position"] for x in windows}
        for x in statuses:
            status_counts[x["frozen_status"]] += 1
            e.require(
                x["query_position"] == by_anchor.get(x["anchor_seconds"])
                and (x["frozen_status"] == "eligible") == (x["query_position"] is not None),
                "Status/window association mismatch",
            )
        for x in windows:
            anchor = x["anchor_seconds"]
            e.require(
                x["history_start_seconds"] == anchor - 300
                and x["history_end_seconds"] == anchor
                and x["horizon_start_exclusive_seconds"] == anchor
                and x["horizon_end_inclusive_seconds"] == anchor + 300
                and x["future_observation_end_seconds"] == anchor + 360
                and x["future_observation_end_seconds"] <= entry["duration_seconds"],
                "Forecast temporal contract mismatch",
            )
            onset = x["ground_truth"]["matched_episode_onset_seconds"]
            e.require((onset is None) == (x["ground_truth"]["label"] == 0), "Label/event mismatch")
            if onset is not None:
                e.require(
                    anchor < onset <= anchor + 300
                    and any(ev["onset_seconds"] == onset for ev in events),
                    "Event/window join mismatch",
                )
        if context is not None:
            case = next(c for c in context["cases"] if c["ordinal"] == ordinal)
            source_rows = [x for x in context["windows"] if x["case_id"] == case["original"]]
            offset = float(case["clinical"].opstart)
            expected = [e.window_record(x, offset, context["predictions"]) for x in source_rows]
            e.require(windows == expected, "Frozen probabilities/score/window routing differ")
            expected_events = e.event_records(
                [x for x in context["episodes"] if x["case_id"] == case["original"]], offset
            )
            e.require(events == expected_events, "Frozen event routing differs")
            expected_statuses = e.anchor_records(
                [x for x in context["anchors"] if x["case_id"] == case["original"]],
                offset,
                source_rows,
            )
            e.require(statuses == expected_statuses, "Frozen status routing differs")
        rows.extend(windows)
        episodes += len(events)
        evaluable += sum(x["evaluable"] for x in events)
        positives += sum(x["ground_truth"]["label"] for x in windows)
    positions = sorted(x["query_position"] for x in rows)
    e.require(
        positions == list(range(3146)) and positives == 116, "Global frozen counts/order mismatch"
    )
    if context is not None:
        e.require(
            manifest["source_registry_sha256"]
            == context["binding"]["receipt"]["source_registry_sha256"],
            "Registry provenance differs",
        )
        e.require(
            manifest["models"] == e.model_definitions(context["binding"]["lock"]),
            "Locked model presentation definitions differ",
        )
    for name in e.CHANNELS:
        c = channel_totals[name]
        c["valid_fraction"] = c["valid_seconds"] / c["grid_seconds"]
    return {
        "counts": {
            "cases": len(entries),
            "windows": len(rows),
            "positives": positives,
            "episodes": episodes,
            "evaluable_episodes": evaluable,
            "scheduled_anchor_status": dict(status_counts),
            "public_files": len(hashes),
        },
        "coverage": {name: dict(channel_totals[name]) for name in e.CHANNELS},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=e.FINAL)
    args = parser.parse_args()
    e.install_guard()
    state = e.tree_state()
    context = e.load_context(e.bind_sources())
    summary = validate_export(args.root, context)
    e.verify_unchanged(context["binding"]["bound"], state)
    print(e.canonical({"status": "VALIDATED", **summary}).decode(), end="")


if __name__ == "__main__":
    main()
