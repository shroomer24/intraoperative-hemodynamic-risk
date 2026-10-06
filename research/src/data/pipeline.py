"""Local ingestion through auditable tables and pilot reports; no model training."""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from intraop.config import DatasetConfig, PathsConfig, ProjectConfig, WindowsConfig
from intraop.data.acquisition import METADATA_FILES, SOURCE_URL, sha256_file, source_checksums
from intraop.data.datasets import METADATA_COLUMNS, ClassificationDataset
from intraop.data.labels import (
    ANCHOR_COLUMNS,
    EPISODE_COLUMNS,
    detect_episodes,
    label_anchors,
    mark_evaluable_episodes,
)
from intraop.data.preprocessing import preprocess_case
from intraop.data.protocol import PROTOCOL_VERSION, TRACKS
from intraop.data.vitaldb_reader import (
    READER_VERSION,
    load_clinical_metadata,
    population_eligibility,
    read_local_case,
)
from intraop.evaluation.subject_splits import (
    make_subject_split,
    partition_positions,
    persist_split_manifest,
)
from intraop.features.base import Window
from intraop.features.core import FEATURE_NAMES, CoreFeatureExtractor
from intraop.reproducibility import write_run_manifest


def _source_provenance(root: Path, case_ids: list[int]) -> dict:
    required = ("clinical_data.csv", "track_names.csv", "LICENSE.txt")
    if any(not (root / name).is_file() for name in required):
        raise FileNotFoundError(
            "Local clinical_data.csv, track_names.csv, and LICENSE.txt are required"
        )
    checksums = source_checksums(root) if (root / "SHA256SUMS.txt").exists() else {}
    names = [name for name in METADATA_FILES if (root / name).is_file()]
    names += [
        f"vital_files/{case:04d}.vital"
        for case in case_ids
        if (root / f"vital_files/{case:04d}.vital").is_file()
    ]
    files = []
    for name in names:
        path = root / name
        digest = sha256_file(path)
        if checksums and name != "SHA256SUMS.txt" and checksums.get(name) != digest:
            raise ValueError(f"Canonical release checksum mismatch: {name}")
        files.append(
            {"path": name, "url": SOURCE_URL + name, "bytes": path.stat().st_size, "sha256": digest}
        )
    return {
        "source_url": SOURCE_URL,
        "version": "1.0.0",
        "files": files,
        "reader_version": READER_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "release_checksums_verified": bool(checksums),
        "source_assertion": "Local PhysioNet v1.0.0 copy supplied by acquisition/user",
    }


@dataclass(frozen=True)
class CaseTables:
    dataset: ClassificationDataset
    episodes: pd.DataFrame
    anchor_audit: pd.DataFrame
    reports: list[dict]


def build_case_tables(root: Path, selected: pd.DataFrame) -> CaseTables:
    """One frozen reader/processing/label/feature implementation for every run."""
    feature_rows, metadata_rows, anchor_tables, episode_tables, reports = [], [], [], [], []
    extractor = CoreFeatureExtractor()
    for _, row in selected.iterrows():
        case_id, subject_id = str(int(row.caseid)), str(int(row.subjectid))
        report = {
            "case_id": case_id,
            "subject_id": subject_id,
            "parsed": False,
            "included": False,
            "exclusion_reason": None,
            "anomaly": None,
        }
        reason = population_eligibility(row)
        if reason:
            report["exclusion_reason"] = reason
            reports.append(report)
            continue
        try:
            signals = read_local_case(root, row)
        except (OSError, ValueError, RuntimeError, ImportError) as error:
            report["anomaly"] = f"{type(error).__name__}: {error}"
            report["exclusion_reason"] = "parse_error"
            reports.append(report)
            continue
        report.update({"parsed": True, "reader_audit": signals.audit})
        for signal in TRACKS:
            report[f"{signal}_available"] = signals.audit["tracks"][signal]["track_present"]
        if not signals.audit["tracks"]["map"]["track_present"] or signals.samples["map"].empty:
            report["exclusion_reason"] = "required_art_mbp_unavailable"
            reports.append(report)
            continue
        case = preprocess_case(signals)
        episodes = detect_episodes(case)
        anchors = label_anchors(case, episodes)
        episodes = mark_evaluable_episodes(episodes, anchors)
        eligible = anchors.loc[anchors.status == "eligible"]
        for _, anchor in eligible.iterrows():
            t = float(anchor.anchor_time_seconds)
            history = case.grid.loc[
                (case.grid.time_seconds > t - 300) & (case.grid.time_seconds <= t)
            ].copy()
            raw = {
                signal: samples.loc[
                    (samples.time_seconds > t - 300) & (samples.time_seconds <= t)
                ].copy()
                for signal, samples in case.raw_valid_samples.items()
            }
            window = Window(
                case.case_id,
                case.subject_id,
                t,
                t - 300,
                t,
                history,
                raw,
                future_observation_end_seconds=float(anchor.future_observation_end_seconds),
            )
            feature_rows.append(extractor.extract(window))
            metadata_rows.append(anchor.to_dict())
        anchor_tables.append(anchors)
        episode_tables.append(episodes)
        report.update(
            {
                "included": True,
                "eligible_anchors": len(eligible),
                "confirmed_events": len(episodes),
                "evaluable_events": int(episodes.evaluable.sum()),
                "positives": int(eligible.label.sum()),
                "censored_anchors": int((anchors.status == "censored").sum()),
                "ineligible_anchors": int((anchors.status == "ineligible").sum()),
                "surgical_interval_seconds": case.opend_seconds - case.opstart_seconds,
                "signal_audit": case.audit,
            }
        )
        reports.append(report)
    anchor_audit = (
        pd.concat(anchor_tables, ignore_index=True)
        if anchor_tables
        else pd.DataFrame(columns=ANCHOR_COLUMNS)
    )
    nonempty_episode_tables = [table for table in episode_tables if not table.empty]
    episodes = (
        pd.concat(nonempty_episode_tables, ignore_index=True)
        if nonempty_episode_tables
        else pd.DataFrame(columns=EPISODE_COLUMNS)
    )
    features = pd.DataFrame(feature_rows, columns=FEATURE_NAMES, dtype=float)
    metadata_frame = pd.DataFrame(metadata_rows, columns=ANCHOR_COLUMNS)
    labels = metadata_frame.pop("label").astype("int8").rename("label")
    metadata_frame = metadata_frame.drop(columns=["status", "reason"])
    for name in METADATA_COLUMNS[2:]:
        metadata_frame[name] = metadata_frame[name].astype(float)
    for frame in (features, labels, metadata_frame):
        frame.index.name = "window_id"
    dataset = ClassificationDataset(features, labels, metadata_frame)
    return CaseTables(dataset, episodes, anchor_audit, reports)


def run_tables(
    root: Path,
    output: Path,
    case_ids: list[int],
    *,
    seed: int = 42,
    realized_development_subjects: list[str] | None = None,
) -> dict:
    """Generate every eligible row with separate pilot/realized split policy.

    Defaults retain the original candidate-population pilot reservation. An
    explicit pilot subject list selects the actual eligible-subject population.
    Both policies use the same frozen build_case_tables implementation.
    """
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("Supply distinct case IDs explicitly")
    provenance = _source_provenance(root, case_ids)
    package_root = Path(__file__).resolve().parents[1]
    implementation = sorted(package_root.rglob("*.py"))
    provenance["implementation_sha256"] = {
        str(path.relative_to(package_root)): sha256_file(path) for path in implementation
    }
    repository = package_root.parent
    protocol_document = repository / "docs/research_protocol_supplied.md"
    provenance["supplied_protocol_sha256"] = (
        sha256_file(protocol_document) if protocol_document.exists() else None
    )
    clinical = load_clinical_metadata(root)
    selected = clinical.loc[clinical.caseid.isin(case_ids)].sort_values("caseid")
    if len(selected) != len(case_ids):
        raise ValueError("A requested case is absent from clinical metadata")
    population_reasons = clinical.apply(population_eligibility, axis=1)
    candidates = clinical.loc[population_reasons.isna()]
    tables = build_case_tables(root, selected)
    dataset, episodes, anchor_audit, reports = (
        tables.dataset,
        tables.episodes,
        tables.anchor_audit,
        tables.reports,
    )
    labels = dataset.y
    if realized_development_subjects is None:
        development = sorted(
            {
                str(int(row.subjectid))
                for _, row in selected.iterrows()
                if population_eligibility(row) is None
            }
        )
        subjects = candidates.subjectid.astype(int).astype(str).tolist()
        episode_status = None
        split_population = "clinical_candidates"
    else:
        development = realized_development_subjects
        subjects = sorted(dataset.groups.unique())
        positive = set(episodes.loc[episodes.evaluable.astype(bool), "subject_id"])
        episode_status = {subject: subject in positive for subject in subjects}
        split_population = "actual_eligible_window_cohort"
    manifest = make_subject_split(
        subjects, seed=seed, episode_status=episode_status, development_subjects=development
    )
    manifest["population_policy"] = split_population
    manifest["clinical_metadata_sha256"] = sha256_file(root / "clinical_data.csv")
    partitions = partition_positions(dataset.metadata, manifest)
    output.mkdir(parents=True, exist_ok=True)
    split_name = (
        "split_manifest.json"
        if realized_development_subjects is None
        else "realized_split_manifest.json"
    )
    persist_split_manifest(output / split_name, manifest)
    dataset.X.to_csv(output / "features.csv", index=False)
    dataset.y.to_csv(output / "labels.csv", index=False)
    dataset.metadata.to_csv(output / "metadata.csv")
    episodes.to_csv(output / "episodes.csv", index=False)
    anchor_audit.to_csv(output / "anchor_audit.csv", index=False)
    pd.DataFrame(
        [{k: v for k, v in report.items() if not isinstance(v, dict)} for report in reports]
    ).to_csv(output / "case_report.csv", index=False)
    schema = {
        "version": PROTOCOL_VERSION,
        "feature_count": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "units": "See docs/feature_schema.md",
    }
    (output / "feature_schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    included = [r for r in reports if r["included"]]
    seconds = sum(r["signal_audit"]["surgical_grid_seconds"] for r in included)
    track_valid = {
        signal: sum(r["signal_audit"]["valid_seconds"][signal] for r in included)
        for signal in TRACKS
    }
    subject_rows = dict(Counter(dataset.groups))
    cohort = {
        "protocol_version": PROTOCOL_VERSION,
        "cases_attempted": len(case_ids),
        "cases_parsed": sum(r["parsed"] for r in reports),
        "cases_included": len(included),
        "eligible_windows": len(labels),
        "positive_windows": int(labels.sum()),
        "negative_windows": int((labels == 0).sum()),
        "positive_rate": float(labels.mean()) if len(labels) else None,
        "unique_subjects": dataset.groups.nunique(),
        "unique_cases": dataset.metadata.case_id.nunique(),
        "unique_positive_subjects": dataset.metadata.loc[labels == 1, "subject_id"].nunique(),
        "confirmed_hypotensive_episodes": len(episodes),
        "evaluable_episodes": int(episodes.evaluable.sum()),
        "episodes_without_forecast_opportunity": int((~episodes.evaluable.astype(bool)).sum()),
        "censored_candidate_anchors": int((anchor_audit.status == "censored").sum()),
        "ineligible_candidate_anchors": int((anchor_audit.status == "ineligible").sum()),
        "anchor_status_reasons": dict(Counter(anchor_audit.reason[anchor_audit.reason != ""])),
        "surgical_interval_hours": sum(r["surgical_interval_seconds"] for r in included) / 3600,
        "monitored_hours": track_valid["map"] / 3600,
        "map_rejected_surgical_records": sum(
            r["signal_audit"]["rejected_surgical_records"]["map"] for r in included
        ),
        "map_ordering_inconsistent_seconds": sum(
            r["signal_audit"]["map_ordering_inconsistent_seconds"] for r in included
        ),
        "map_ordering_auditable_seconds": sum(
            r["signal_audit"]["map_ordering_auditable_seconds"] for r in included
        ),
        "track_coverage": {
            signal: {
                "valid_seconds": track_valid[signal],
                "surgical_grid_seconds": seconds,
                "valid_fraction": track_valid[signal] / seconds if seconds else None,
                "missing_fraction": 1 - track_valid[signal] / seconds if seconds else None,
            }
            for signal in TRACKS
        },
        "track_availability_cases": {
            signal: sum(r.get(f"{signal}_available", False) for r in reports) for signal in TRACKS
        },
        "windows_per_subject": subject_rows,
        "events_per_subject": {
            s: int((episodes.subject_id == s).sum())
            for s in sorted({r["subject_id"] for r in included})
        },
        "evaluable_events_per_subject": {
            s: int(((episodes.subject_id == s) & episodes.evaluable.astype(bool)).sum())
            for s in sorted({r["subject_id"] for r in included})
        },
        "population_exclusions_all_clinical_cases": dict(Counter(population_reasons.dropna())),
        "pilot_case_exclusions": dict(
            Counter(r["exclusion_reason"] for r in reports if r["exclusion_reason"])
        ),
        "split_subject_counts": manifest["counts"],
        "pilot_windows_by_partition": {p: len(rows) for p, rows in partitions.items()},
        "split_population": split_population,
        "split_stratification_fallback_reason": manifest["stratification_fallback_reason"],
        "cases": reports,
        "limitations": [
            "Selected arterial-line pilot is a development smoke test; "
            "its prevalence is not representative.",
            "All inspected pilot subjects are pinned to training; no test performance is measured.",
            "Candidate-population quotas are provisional; report realized "
            "ART-cohort subject quotas before modeling.",
            "Numeric sample-and-hold does not prove uninterrupted beat-level hypotension.",
            "Waveform adjudication is pending; MAP ordering flags do not censor records.",
            "No calendar temporal holdout is possible from relative times/randomized case IDs.",
        ],
    }
    (output / "cohort_report.json").write_text(json.dumps(cohort, indent=2) + "\n")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    report_lines = [
        "# VitalDB v0.1 development pilot",
        "",
        "No model was trained or evaluated.",
        "",
    ]
    for name in (
        "cases_attempted",
        "cases_parsed",
        "cases_included",
        "eligible_windows",
        "positive_windows",
        "negative_windows",
        "positive_rate",
        "unique_subjects",
        "unique_cases",
        "confirmed_hypotensive_episodes",
        "evaluable_episodes",
        "censored_candidate_anchors",
        "monitored_hours",
        "map_rejected_surgical_records",
        "map_ordering_inconsistent_seconds",
    ):
        report_lines.append(f"- {name}: {cohort[name]}")
    report_lines += [
        "",
        "## Case parsing and outcomes",
        "",
        "| Case | Parsed | MAP | Eligible | Events | Positive | Censored | Anomaly/exclusion |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        report_lines.append(
            f"| {r['case_id']} | {r['parsed']} | {r.get('map_available')} | "
            f"{r.get('eligible_anchors', 0)} | {r.get('confirmed_events', 0)} | "
            f"{r.get('positives', 0)} | {r.get('censored_anchors', 0)} | "
            f"{r.get('anomaly') or r.get('exclusion_reason') or ''} |"
        )
    report_lines += [
        "",
        "## Signal coverage",
        "",
        "| Signal | Cases available | Valid fraction | Missing fraction |",
        "|---|---|---|---|",
    ]
    for signal, coverage in cohort["track_coverage"].items():
        report_lines.append(
            f"| {signal} | {cohort['track_availability_cases'][signal]} | "
            f"{coverage['valid_fraction']} | {coverage['missing_fraction']} |"
        )
    report_lines += [
        "",
        "## Split scope and limitations",
        "",
        f"Candidate subject counts: {manifest['counts']}",
        f"Pilot window counts: {cohort['pilot_windows_by_partition']}",
        f"Stratification fallback: {manifest['stratification_fallback_reason']}",
        "",
    ]
    report_lines += [f"- {limitation}" for limitation in cohort["limitations"]]
    report_lines += [
        "",
        "## Downloaded/source files",
        "",
        "| File | Bytes | SHA256 |",
        "|---|---|---|",
    ]
    report_lines += [f"| {f['path']} | {f['bytes']} | {f['sha256']} |" for f in provenance["files"]]
    (output / "cohort_report.md").write_text("\n".join(report_lines) + "\n")
    resolved_output = output.resolve()
    runtime_config = ProjectConfig(
        paths=PathsConfig(
            raw=root.resolve(),
            interim=resolved_output,
            processed=resolved_output,
            artifacts=resolved_output,
        ),
        seed=seed,
        dataset=DatasetConfig(
            case_id_column="case_id",
            subject_id_column="subject_id",
            time_seconds_column="time_seconds",
            signal_columns=tuple(TRACKS),
        ),
        windows=WindowsConfig(lookback_seconds=300, stride_seconds=60),
    )
    write_run_manifest(output / "runtime_manifest.json", runtime_config)
    (output / "table_manifest.json").write_text(
        json.dumps(
            {
                "protocol_version": PROTOCOL_VERSION,
                "seed": seed,
                "tables": {
                    name: sha256_file(output / name)
                    for name in (
                        "features.csv",
                        "labels.csv",
                        "metadata.csv",
                        "episodes.csv",
                        "anchor_audit.csv",
                    )
                },
            },
            indent=2,
        )
        + "\n"
    )
    return cohort


def run_pilot(root: Path, output: Path, case_ids: list[int], *, seed: int = 42) -> dict:
    """Retained pilot interface; all inspected subjects stay training-only."""
    return run_tables(root, output, case_ids, seed=seed)
