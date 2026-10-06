"""Read-only human inspection of boundaries and raw arterial records; never predictors."""

from pathlib import Path

import numpy as np
import pandas as pd

from intraop.data.preprocessing import preprocess_case, valid_samples
from intraop.data.vitaldb_reader import read_local_case

WAVEFORM_TRACK = "SNUADC/ART"
BOUNDARY_TOLERANCE_SECONDS = 60


def _range(values):
    return (float(values.min()), float(values.max())) if len(values) else (np.nan, np.nan)


def boundary_observation(signals, row) -> dict:
    """Observe frozen converted times without modifying signals or boundaries.

    Monitor envelope is the six frozen numeric tracks, not every optional device.
    Nonoverlap and >60s discrepancies are warnings, never inferred corrections.
    """
    times = pd.concat([s.time_seconds for s in signals.samples.values()], ignore_index=True)
    first, last = _range(times)
    map_first, map_last = _range(valid_samples(signals.samples["map"], signal="map").time_seconds)
    overlap = bool(len(times) and max(first, row.opstart) <= min(last, row.opend, row.caseend))
    warnings = []
    if not overlap:
        warnings.append("surgery_does_not_overlap_numeric_monitor")
    if row.opstart < -BOUNDARY_TOLERANCE_SECONDS:
        warnings.append("opstart_materially_before_recording_zero")
    if row.opend > row.caseend + BOUNDARY_TOLERANCE_SECONDS:
        warnings.append("opend_materially_after_caseend")
    if len(times):
        if first < -BOUNDARY_TOLERANCE_SECONDS or last > row.caseend + BOUNDARY_TOLERANCE_SECONDS:
            warnings.append("numeric_monitor_outside_recording_bounds")
        if first > row.opstart + BOUNDARY_TOLERANCE_SECONDS:
            warnings.append("surgery_starts_before_numeric_monitor")
        if last < min(row.opend, row.caseend) - BOUNDARY_TOLERANCE_SECONDS:
            warnings.append("surgery_ends_after_numeric_monitor")
    if np.isfinite(map_first) and map_first > row.opstart + BOUNDARY_TOLERANCE_SECONDS:
        warnings.append("map_starts_after_surgery_start")
    if (
        np.isfinite(map_last)
        and map_last < min(row.opend, row.caseend) - BOUNDARY_TOLERANCE_SECONDS
    ):
        warnings.append("map_ends_before_surgery_end")
    return {
        "case_id": signals.case_id,
        "subject_id": signals.subject_id,
        "recording_origin_source_clock": signals.audit.get("source_clock_origin"),
        "recording_origin_relative_seconds": 0.0,
        "origin_method": signals.audit.get("origin_method"),
        "first_packet_relative_seconds": signals.audit.get("first_packet_relative_seconds"),
        "last_packet_relative_seconds": signals.audit.get("last_packet_relative_seconds"),
        "monitor_tracks": "six_frozen_numeric_tracks",
        "first_monitor_seconds": first,
        "last_monitor_seconds": last,
        "opstart_seconds": float(row.opstart),
        "opend_seconds": float(row.opend),
        "caseend_seconds": float(row.caseend),
        "first_usable_map_seconds": map_first,
        "last_usable_map_seconds": map_last,
        "surgery_monitor_overlap": overlap,
        "warnings": ";".join(warnings),
        "parser_error": "",
        "plot_filename": "",
    }


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _boundary_plot(signals, row, path):
    plt = _pyplot()
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    data = signals.samples["map"]
    axes[0].plot(data.time_seconds, data.value, linewidth=0.6, label="original numeric ART_MBP")
    axes[0].axhline(65, color="red", linestyle="--", label="65 mmHg")
    axes[0].set_ylabel("MAP (mmHg)")
    axes[0].legend(loc="upper right")
    for ax in axes:
        for name, value, color in (
            ("recording zero", 0, "black"),
            ("opstart", row.opstart, "green"),
            ("opend", row.opend, "orange"),
            ("caseend", row.caseend, "purple"),
        ):
            ax.axvline(value, label=name, color=color, linestyle="--", linewidth=1)
    axes[1].hlines(2, row.opstart, row.opend, color="green", linewidth=5, label="surgery")
    first, last = _range(pd.concat([s.time_seconds for s in signals.samples.values()]))
    axes[1].hlines(1, first, last, color="blue", linewidth=5, label="numeric monitor")
    map_first, map_last = _range(valid_samples(data, signal="map").time_seconds)
    axes[1].hlines(0, map_first, map_last, color="red", linewidth=5, label="usable MAP envelope")
    axes[1].set_yticks([0, 1, 2], ["MAP envelope", "monitor", "surgery"])
    axes[1].set_xlabel("Seconds from recording origin (no calendar conversion)")
    axes[1].legend(loc="upper right", ncol=4)
    fig.suptitle(f"Boundary audit — case {signals.case_id}; offsets retained")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def generate_boundary_audit(root: Path, output: Path, clinical, case_ids) -> pd.DataFrame:
    directory = output / "boundary_audit"
    (directory / "suspicious").mkdir(parents=True, exist_ok=True)
    rows = []
    for case_id in sorted(case_ids):
        row = clinical.loc[clinical.caseid == case_id].iloc[0]
        try:
            signals = read_local_case(root, row)
            result = boundary_observation(signals, row)
            if result["warnings"]:
                name = f"suspicious/case_{case_id:04d}.png"
                _boundary_plot(signals, row, directory / name)
                result["plot_filename"] = name
        except (OSError, ValueError, RuntimeError, ImportError) as error:
            result = {
                "case_id": str(case_id),
                "subject_id": str(int(row.subjectid)),
                "opstart_seconds": row.opstart,
                "opend_seconds": row.opend,
                "caseend_seconds": row.caseend,
                "parser_error": f"{type(error).__name__}: {error}",
                "warnings": "parser_failure_no_timestamp_repair",
                "plot_filename": "",
            }
        rows.append(result)
        print(f"Boundary checked {len(rows)}/{len(case_ids)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(directory / "boundary_manifest.csv", index=False)
    return frame


def waveform_segment(track, origin: float, start: float, end: float):
    """Original record dt + sample index/srate, with documented integer gain/offset.

    No resampling, interpolation, morphology or label decisions. NaN separators
    prevent plot lines from bridging packet gaps/overlaps or reordered records.
    """
    if track is None:
        return np.array([]), np.array([])
    if track.type != 1 or not np.isfinite(track.srate) or track.srate <= 0:
        raise ValueError("Expected waveform with finite positive sample rate")
    times, values = [], []
    for record in track.recs:
        dt = float(record["dt"]) - origin
        count = len(record["val"])
        first = max(0, int(np.ceil((start - dt) * track.srate)))
        last = min(count, int(np.floor((end - dt) * track.srate)) + 1)
        if last <= first:
            continue
        x = dt + np.arange(first, last, dtype=float) / track.srate
        y = np.asarray(record["val"][first:last], dtype=float)
        if track.fmt > 2:
            y = y * track.gain + track.offset
        times.extend((x, np.array([np.nan])))
        values.extend((y, np.array([np.nan])))
    return (
        (np.concatenate(times), np.concatenate(values)) if times else (np.array([]), np.array([]))
    )


def _diverse(frame, number, used):
    """Deterministic ranked round-robin across cases; skip previously used anchors."""
    pools = [group.to_dict("records") for _, group in frame.groupby("case_id", sort=True)]
    result = []
    while pools and len(result) < number:
        remaining = []
        for pool in pools:
            while pool:
                row = pool.pop(0)
                key = (row["case_id"], row["anchor_time_seconds"])
                if key not in used:
                    result.append(row)
                    used.add(key)
                    break
            if pool:
                remaining.append(pool)
            if len(result) == number:
                break
        pools = remaining
    return result


def select_waveform_examples(anchors, subjects, minimum_recent_map) -> list[dict]:
    """Training-only 10 onsets, 5 difficult positives, 5 gray/negative anchors.

    Difficulty ranks ordering flags, then closeness to 65 in the last minute,
    then history coverage. This affects audit figures only, never acquisition/X.
    """
    frame = anchors.loc[(anchors.subject_id.isin(subjects)) & (anchors.status == "eligible")].copy()
    frame["recent_map_minimum"] = [
        minimum_recent_map[(r.case_id, r.anchor_time_seconds)] for r in frame.itertuples()
    ]
    positive = frame.loc[frame.label == 1].sort_values(["case_id", "anchor_time_seconds"])
    onset = positive.drop_duplicates(["case_id", "matched_episode_onset_seconds"])
    difficult = positive.sort_values(
        [
            "map_ordering_inconsistent_history_seconds",
            "recent_map_minimum",
            "history_map_coverage",
            "anchor_time_seconds",
        ],
        ascending=[False, True, True, True],
    )
    negative = frame.loc[frame.label == 0].sort_values(
        ["recent_map_minimum", "anchor_time_seconds"]
    )
    used, selected = set(), []
    for category, candidates, number in (
        ("confirmed_onset", onset, 10),
        ("borderline_positive", difficult, 5),
        ("negative_or_gray", negative, 5),
    ):
        for row in _diverse(candidates, number, used):
            row["category"] = category
            selected.append(row)
    return selected


def generate_waveform_audit(root: Path, output: Path, clinical, pilot_cases, pilot_subjects):
    from vitaldb import VitalFile

    directory = output / "waveform_audit"
    directory.mkdir(parents=True, exist_ok=True)
    anchors = pd.read_csv(output / "anchor_audit.csv", dtype={"case_id": str, "subject_id": str})
    cache, minima = {}, {}
    for case_id in sorted(pilot_cases):
        row = clinical.loc[clinical.caseid == case_id].iloc[0]
        signals = read_local_case(root, row)
        case = preprocess_case(signals)
        cache[signals.case_id] = (signals, case, row)
        for anchor in anchors.loc[anchors.case_id == signals.case_id].itertuples():
            t = anchor.anchor_time_seconds
            minima[(signals.case_id, t)] = case.grid.loc[
                (case.grid.time_seconds > t - 60) & (case.grid.time_seconds <= t), "map"
            ].min()
    examples = select_waveform_examples(anchors, set(pilot_subjects), minima)
    tracks, rows = {}, []
    plt = _pyplot()
    for index, example in enumerate(examples):
        cid = example["case_id"]
        signals, case, clinical_row = cache[cid]
        if cid not in tracks:
            vital = VitalFile(
                str((root / "vital_files" / f"{int(cid):04d}.vital").resolve()),
                track_names=[WAVEFORM_TRACK],
            )
            tracks[cid] = vital.trks.get(WAVEFORM_TRACK)
        track = tracks[cid]
        t, onset = example["anchor_time_seconds"], example["matched_episode_onset_seconds"]
        focus = onset if np.isfinite(onset) else t
        start, end = max(0, t - 180), max(t + 360, focus + 90)
        x, y = waveform_segment(track, signals.audit["source_clock_origin"], start, end)
        name = f"{index + 1:02d}_{example['category']}_case_{int(cid):04d}_t{int(t)}.png"
        if not np.isfinite(y).any() and not example.get("replacement_for_plot"):
            # Keep explicit missing-waveform evidence and supplement it when
            # practical. Availability affects human audits only, never labels/X.
            used = {(r["case_id"], r["anchor_time_seconds"]) for r in examples}
            represented = {r["case_id"] for r in examples}
            candidates = anchors.loc[
                anchors.subject_id.isin(pilot_subjects)
                & (anchors.status == "eligible")
                & (anchors.label == example["label"])
                & (anchors.case_id != cid)
            ].copy()
            candidates["represented_case"] = candidates.case_id.isin(represented)
            candidates = candidates.sort_values(
                ["represented_case", "case_id", "anchor_time_seconds"]
            )
            for candidate in candidates.to_dict("records"):
                key = (candidate["case_id"], candidate["anchor_time_seconds"])
                if key in used:
                    continue
                other_signals, _, _ = cache[candidate["case_id"]]
                if candidate["case_id"] not in tracks:
                    other_vital = VitalFile(
                        str(
                            (
                                root / "vital_files" / f"{int(candidate['case_id']):04d}.vital"
                            ).resolve()
                        ),
                        track_names=[WAVEFORM_TRACK],
                    )
                    tracks[candidate["case_id"]] = other_vital.trks.get(WAVEFORM_TRACK)
                _, other_y = waveform_segment(
                    tracks[candidate["case_id"]],
                    other_signals.audit["source_clock_origin"],
                    max(0, key[1] - 180),
                    key[1] + 360,
                )
                if np.isfinite(other_y).any():
                    candidate.update(
                        category=example["category"],
                        replacement_for_plot=name,
                        recent_map_minimum=minima[key],
                    )
                    examples.append(candidate)
                    break
        fig, axes = plt.subplots(3, 1, figsize=(13, 8), constrained_layout=True)
        raw = signals.samples["map"]
        raw = raw.loc[raw.time_seconds.between(start, end)]
        grid = case.grid.loc[case.grid.time_seconds.between(start, end)]
        axes[0].plot(
            raw.time_seconds, raw.value, ".", markersize=2, label="original ART_MBP records"
        )
        axes[0].step(
            grid.time_seconds, grid["map"], where="post", linewidth=1, label="frozen causal MAP"
        )
        axes[0].axhline(65, color="red", linestyle="--", label="65 mmHg")
        axes[0].legend(loc="upper right")
        axes[1].plot(x, y, linewidth=0.3, color="#4865a2")
        axes[1].set_title(
            "Raw SNUADC/ART packets — no interpolation"
            if len(x)
            else "SNUADC/ART unavailable in this interval"
        )
        zoom_x, zoom_y = waveform_segment(
            track, signals.audit["source_clock_origin"], focus - 4, focus + 4
        )
        axes[2].plot(zoom_x, zoom_y, linewidth=0.8, color="#4865a2")
        axes[2].set_title("8-second raw waveform detail around onset/anchor")
        axes[2].set_xlim(focus - 4, focus + 4)
        for ax in axes:
            ax.set_ylabel("mmHg")
            ax.set_xlabel("Seconds from recording origin")
            ax.ticklabel_format(style="plain", axis="x", useOffset=False)
            bounds = (focus - 4, focus + 4) if ax is axes[2] else (start, end)
            if ax is not axes[2]:
                ax.set_xlim(start, end)
            for label, value, color in (
                ("anchor t", t, "black"),
                ("onset s", onset, "red"),
                ("opstart", clinical_row.opstart, "green"),
                ("opend", clinical_row.opend, "orange"),
            ):
                if np.isfinite(value) and bounds[0] <= value <= bounds[1]:
                    ax.axvline(value, color=color, linestyle="--", linewidth=1)
                    height = {"anchor t": 0.96, "onset s": 0.80, "opstart": 0.64, "opend": 0.48}
                    ax.text(
                        value,
                        height[label],
                        label,
                        transform=ax.get_xaxis_transform(),
                        fontsize=8,
                        color=color,
                    )
        fig.suptitle(
            f"Human audit only — {example['category']}, case {cid}\n"
            f"opstart={clinical_row.opstart:g}, opend={clinical_row.opend:g}; "
            "no clinical adjudication"
        )
        fig.savefig(directory / name, dpi=150)
        plt.close(fig)
        packet_times = (
            np.array([r["dt"] for r in track.recs]) if track is not None else np.array([])
        )
        observations = ["numeric label unchanged; human review required"]
        if not len(x):
            observations.append("no waveform samples in plotted interval")
        if example["map_ordering_inconsistent_history_seconds"]:
            observations.append("numeric SBP/MAP/DBP ordering disagreement in history")
        rows.append(
            {
                **example,
                "anchor": t,
                "onset": onset,
                "plot_filename": name,
                "partition": "training",
                "waveform_track": WAVEFORM_TRACK,
                "waveform_track_present": track is not None,
                "waveform_available": bool(np.isfinite(y).any()),
                "waveform_finite_samples_in_plot": int(np.isfinite(y).sum()),
                "waveform_plot_start_seconds": start,
                "waveform_plot_end_seconds": end,
                "waveform_first_sample_in_plot_seconds": float(np.nanmin(x)) if len(x) else np.nan,
                "waveform_last_sample_in_plot_seconds": float(np.nanmax(x)) if len(x) else np.nan,
                "waveform_onset_detail_available": bool(np.isfinite(zoom_y).any())
                if np.isfinite(onset)
                else None,
                "waveform_sample_rate": track.srate if track is not None else np.nan,
                "waveform_gain": track.gain if track is not None else np.nan,
                "waveform_offset": track.offset if track is not None else np.nan,
                "waveform_format": track.fmt if track is not None else None,
                "waveform_out_of_order_packets": int((np.diff(packet_times) < 0).sum()),
                "waveform_duplicate_packet_times": len(packet_times) - len(np.unique(packet_times)),
                "numeric_map_coverage": float(grid["map"].notna().mean()),
                "map_rejected_surgical_records": case.audit["rejected_surgical_records"]["map"],
                "map_original_out_of_order_records": signals.audit["tracks"]["map"][
                    "out_of_order_records"
                ],
                "map_original_duplicate_timestamps": signals.audit["tracks"]["map"][
                    "duplicate_timestamps"
                ],
                "map_original_nonfinite_values": signals.audit["tracks"]["map"]["nonfinite_values"],
                "automated_technical_observations": "; ".join(observations),
            }
        )
        print(f"Waveform plotted {index + 1}/{len(examples)}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(directory / "audit_manifest.csv", index=False)
    return result
