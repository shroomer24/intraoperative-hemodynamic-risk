"""Realized-cohort orchestration/reporting over the single frozen table pipeline."""

import json
import shutil
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from intraop.data.acquisition import sha256_file
from intraop.data.audits import generate_boundary_audit, generate_waveform_audit
from intraop.data.datasets import ClassificationDataset
from intraop.data.pipeline import run_tables
from intraop.data.vitaldb_reader import load_clinical_metadata
from intraop.evaluation.subject_splits import PARTITIONS
from intraop.features.core import FEATURE_NAMES


def load_tables(output):
    X = pd.read_csv(output / "features.csv")
    y = pd.read_csv(output / "labels.csv").label
    metadata = pd.read_csv(
        output / "metadata.csv", index_col="window_id", dtype={"subject_id": str, "case_id": str}
    )
    episodes = pd.read_csv(output / "episodes.csv", dtype={"subject_id": str, "case_id": str})
    if tuple(X.columns) != FEATURE_NAMES:
        raise ValueError("Saved features differ from frozen 74-column whitelist")
    return ClassificationDataset(X, y, metadata), episodes


def _event_concentration(evaluable_by_subject):
    """Describe recurrence concentration without imposing a readiness cutoff."""
    total = sum(evaluable_by_subject.values())
    largest = max(evaluable_by_subject.values(), default=0)
    return {
        "largest_subject_evaluable_episodes": largest,
        "largest_subject_ids": sorted(
            subject
            for subject, count in evaluable_by_subject.items()
            if largest > 0 and count == largest
        ),
        "largest_subject_evaluable_episode_fraction": largest / total if total else None,
    }


def split_statistics(dataset, episodes, manifest, case_reports):
    assignments = manifest["subject_to_partition"]
    if set(assignments) != set(dataset.groups):
        raise ValueError("Realized split must cover exactly eligible subjects")
    groups = {p: {s for s, v in assignments.items() if v == p} for p in PARTITIONS}
    intersections = {
        f"{a} & {b}": sorted(groups[a] & groups[b]) for a, b in combinations(PARTITIONS, 2)
    }
    if any(intersections.values()):
        raise ValueError("Subject overlap")
    pilot = manifest["development_subjects_pinned_to_training"]
    if any(assignments[s] != "training" for s in pilot):
        raise ValueError("Pilot subject leaked outside training")
    counts = {}
    for partition, subjects in groups.items():
        mask = dataset.groups.isin(subjects)
        selected_episodes = episodes.loc[episodes.subject_id.isin(subjects)]
        selected_cases = [r for r in case_reports if r["included"] and r["subject_id"] in subjects]
        evaluable_by_subject = {
            subject: int(
                (
                    (selected_episodes.subject_id == subject)
                    & selected_episodes.evaluable.astype(bool)
                ).sum()
            )
            for subject in sorted(subjects)
        }
        counts[partition] = {
            "subjects": len(subjects),
            "cases": len(
                {
                    r["case_id"]
                    for r in case_reports
                    if r["included"] and r["subject_id"] in subjects
                }
            ),
            "eligible_windows": int(mask.sum()),
            "positive_windows": int(dataset.y[mask].sum()),
            "negative_windows": int((dataset.y[mask] == 0).sum()),
            "monitored_hours": sum(
                r["signal_audit"]["valid_seconds"]["map"] for r in selected_cases
            )
            / 3600,
            "censored_anchors": sum(r["censored_anchors"] for r in selected_cases),
            "evaluable_episodes_by_subject": evaluable_by_subject,
            **_event_concentration(evaluable_by_subject),
            "confirmed_episodes": len(selected_episodes),
            "evaluable_episodes": int(selected_episodes.evaluable.sum()),
            "positive_window_prevalence": float(dataset.y[mask].mean()) if mask.any() else None,
            "positive_subjects": dataset.metadata.loc[
                mask & (dataset.y == 1), "subject_id"
            ].nunique(),
            "subjects_with_evaluable_episodes": selected_episodes.loc[
                selected_episodes.evaluable.astype(bool), "subject_id"
            ].nunique(),
        }
    return {"partitions": counts, "subject_intersections": intersections}


def _distribution(values):
    a = np.asarray(list(values), dtype=float)
    q1, median, q3 = np.percentile(a, [25, 50, 75])
    return {
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def verify_artifacts(output: Path) -> dict:
    dataset, episodes = load_tables(output)
    manifest = json.loads((output / "table_manifest.json").read_text())
    for name, digest in manifest["tables"].items():
        if sha256_file(output / name) != digest:
            raise ValueError(f"Table/audit integrity failure: {name}")
    provenance = json.loads((output / "provenance.json").read_text())
    package_root = Path(__file__).resolve().parents[1]
    for name, digest in provenance["implementation_sha256"].items():
        if sha256_file(package_root / name) != digest:
            raise ValueError(f"Implementation changed since artifact generation: {name}")
    repository = package_root.parent
    for name, digest in provenance["script_sha256"].items():
        if sha256_file(repository / "scripts" / name) != digest:
            raise ValueError(f"Reproduction script changed: {name}")
    if sha256_file(repository / "docs/vitaldb_protocol.md") != provenance["frozen_protocol_sha256"]:
        raise ValueError("Frozen protocol changed")
    split = json.loads((output / "realized_split_manifest.json").read_text())
    report = json.loads((output / "cohort_report.json").read_text())
    statistics = split_statistics(dataset, episodes, split, report["cases"])
    m = dataset.metadata
    if m.duplicated(["case_id", "anchor_time_seconds"]).any():
        raise ValueError("Duplicate saved case/anchor")
    if not (
        (m.history_end_seconds - m.history_start_seconds == 300)
        & (m.future_observation_end_seconds - m.anchor_time_seconds == 360)
        & (m.history_map_coverage >= 0.9)
        & (m.recent_map_valid_seconds == 60)
        & (m.future_map_valid_seconds == 360)
        & (m.anchor_time_seconds % 60 == 0)
    ).all():
        raise ValueError("Saved eligibility/timing integrity failure")
    anchors = pd.read_csv(output / "anchor_audit.csv", dtype={"case_id": str, "subject_id": str})
    eligible = anchors.loc[anchors.status == "eligible"].reset_index(drop=True)
    if len(eligible) != len(m) or not np.array_equal(eligible.label, dataset.y):
        raise ValueError("Saved eligible-anchor/label mismatch")
    for name in ("case_id", "subject_id", "anchor_time_seconds"):
        if not np.array_equal(eligible[name], m[name]):
            raise ValueError(f"Anchor/metadata alignment mismatch: {name}")
    audits = pd.read_csv(output / "waveform_audit/audit_manifest.csv", dtype={"subject_id": str})
    if not set(audits.subject_id) <= set(split["development_subjects_pinned_to_training"]):
        raise ValueError("Waveform audit exposed nonpilot subjects")
    for row in audits.itertuples():
        if not (output / "waveform_audit" / row.plot_filename).is_file():
            raise ValueError("Missing audit figure")
    return {
        "rows": len(dataset.y),
        "predictors": dataset.X.shape[1],
        "table_hashes_verified": True,
        "implementation_hashes_verified": True,
        "eligible_anchor_alignment_verified": True,
        "subject_overlap": False,
        "pilot_training_only": True,
        "forbidden_predictors_absent": True,
        "split_statistics": statistics,
    }


def reuse_waveform_audit(parent: Path, output: Path, root: Path, pilot_cases: list[int]):
    """Reuse verified pilot-only inspection artifacts and their exact raw sources."""
    provenance = json.loads((parent / "provenance.json").read_text())
    source_files = {record["path"]: record["sha256"] for record in provenance["files"]}
    for case in pilot_cases:
        name = f"vital_files/{case:04d}.vital"
        if sha256_file(root / name) != source_files[name]:
            raise ValueError("Waveform audit pilot source changed")
    table = json.loads((parent / "table_manifest.json").read_text())
    audit_files = {
        name: digest
        for name, digest in table["tables"].items()
        if name.startswith("waveform_audit/")
    }
    if "waveform_audit/audit_manifest.csv" not in audit_files:
        raise ValueError("Parent has no verified waveform audit")
    for name, digest in audit_files.items():
        if sha256_file(parent / name) != digest:
            raise ValueError(f"Parent waveform audit changed: {name}")
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent / name, destination)
    return {
        "parent_provenance_sha256": sha256_file(parent / "provenance.json"),
        "files": audit_files,
        "pilot_sources_verified": True,
        "human_adjudication": "pending; existing figures reused without new adjudication",
    }


def run_cohort(
    root: Path,
    output: Path,
    pilot_manifest_path: Path,
    *,
    seed=42,
    boundary=True,
    waveform=True,
    waveform_parent: Path | None = None,
):
    acquisition = json.loads((output / "acquisition_manifest.json").read_text())
    if not acquisition["stop_reason"]:
        raise ValueError("Finish acquisition before freezing the realized cohort")
    cases = acquisition["completed_case_ids"]
    if cases != acquisition["plan"]["candidate_order"][: len(cases)]:
        raise ValueError("Acquired cases are not a deterministic completed prefix")
    if seed != acquisition["plan"]["seed"]:
        raise ValueError("Pipeline seed differs from acquisition plan")
    pilot = json.loads(pilot_manifest_path.read_text())
    development = pilot["development_subjects_pinned_to_training"]
    pilot_cases = acquisition["plan"]["pilot_case_ids"]
    if set(pilot_cases) - set(cases):
        raise ValueError("All ten inspected pilot cases must be retained")
    report = run_tables(root, output, cases, seed=seed, realized_development_subjects=development)
    acquisition["frozen_for_processing"] = True
    (output / "acquisition_manifest.json").write_text(json.dumps(acquisition, indent=2) + "\n")
    dataset, episodes = load_tables(output)
    manifest = json.loads((output / "realized_split_manifest.json").read_text())
    statistics = split_statistics(dataset, episodes, manifest, report["cases"])
    clinical = load_clinical_metadata(root)
    if boundary:
        generate_boundary_audit(root, output, clinical, cases)
    reused_audit = None
    if waveform_parent is not None:
        reused_audit = reuse_waveform_audit(waveform_parent, output, root, pilot_cases)
    elif waveform:
        generate_waveform_audit(root, output, clinical, pilot_cases, development)
    boundaries = pd.read_csv(output / "boundary_audit/boundary_manifest.csv")
    audits = pd.read_csv(output / "waveform_audit/audit_manifest.csv")
    report.pop("pilot_windows_by_partition")
    report["limitations"] = [
        "Deterministic arterial-line-hinted prefix is a selected single-center cohort; "
        "window prevalence is descriptive, not a population estimate.",
        "Repeated anchors/episodes within subjects are correlated; independent subjects "
        "and held-out events limit inference.",
        "Numeric sample-and-hold does not establish uninterrupted beat-level hypotension.",
        "Waveform figures require human adjudication; no clinical true/false calls are automated.",
        "Boundary anomalies are retained/flagged; no unexpected offsets are silently repaired.",
        "Historical treatment changes observed outcomes; no causal treatment-need interpretation.",
        "Relative times and randomized case IDs cannot establish a calendar temporal holdout.",
    ]
    included = [r for r in report["cases"] if r["included"]]
    report.update(
        {
            "candidate_cases_considered": acquisition["candidate_cases_considered"],
            "planned_target_cases": acquisition["plan"]["target_cases"],
            "cases_downloaded_additional": sum(
                not r["initially_local"] for r in acquisition["files"]
            ),
            "cases_available_verified": len(cases),
            "cases_excluded": report["cases_attempted"] - report["cases_included"],
            "case_exclusion_reasons": report["pilot_case_exclusions"],
            "additional_download_bytes": acquisition["transferred_new_vital_bytes"],
            "additional_retained_vital_bytes": sum(
                r["bytes"] for r in acquisition["files"] if not r["initially_local"]
            ),
            "total_selected_vital_bytes": sum(r["bytes"] for r in acquisition["files"]),
            "acquisition_stop_reason": acquisition["stop_reason"],
            "cases_with_art_mbp": report["track_availability_cases"]["map"],
            "cases_with_eligible_anchors": dataset.metadata.case_id.nunique(),
            "unique_eligible_subjects": dataset.groups.nunique(),
            "unique_eligible_cases": dataset.metadata.case_id.nunique(),
            "unique_subjects": len({r["subject_id"] for r in included}),
            "unique_cases": len({r["case_id"] for r in included}),
            "eligible_windows_per_subject_distribution": _distribution(
                report["windows_per_subject"].values()
            ),
            "evaluable_episodes_per_eligible_subject_distribution": _distribution(
                report["evaluable_events_per_subject"].get(s, 0)
                for s in sorted(dataset.groups.unique())
            ),
            "realized_split_statistics": statistics,
            "confirmed_episodes_outside_eligible_subject_split": int(
                (~episodes.subject_id.isin(manifest["subject_to_partition"])).sum()
            ),
            "boundary_audit": {
                "cases_checked": len(boundaries),
                "warning_cases": boundaries.loc[boundaries.warnings.notna(), "case_id"]
                .astype(int)
                .tolist(),
                "parser_error_cases": boundaries.loc[boundaries.parser_error.notna(), "case_id"]
                .astype(int)
                .tolist(),
            },
            "waveform_audit": {
                "examples": len(audits),
                "category_counts": audits.category.value_counts().to_dict(),
                "examples_with_waveform": int(audits.waveform_available.sum()),
                "unique_cases": audits.case_id.nunique(),
                "human_adjudication": "pending; automated observations are technical only",
                "reused_from_parent": reused_audit,
            },
        }
    )
    test = statistics["partitions"]["test"]
    zero_event_partitions = [
        p for p, counts in statistics["partitions"].items() if counts["evaluable_episodes"] == 0
    ]
    report["modeling_assessment"] = {
        "decision": "NO-GO" if zero_event_partitions else "CAUTION",
        "reason": (
            "No evaluable episodes in "
            + ", ".join(zero_event_partitions)
            + "; the four-way event experiment lacks a required denominator."
            if zero_event_partitions
            else "Exploratory hackathon experiments can proceed with stated limitations; "
            "waveform clinical adjudication remains pending. "
            "Independent evaluable-event support (subjects/episodes): "
            + "; ".join(
                f"{p} {c['subjects_with_evaluable_episodes']}/{c['evaluable_episodes']}"
                for p, c in statistics["partitions"].items()
            )
            + ". Recurrent events/windows are correlated; the selected single-center cohort "
            "does not support strong calibration, superiority or clinical claims."
        ),
        "test_evaluable_episodes": test["evaluable_episodes"],
        "test_subjects_with_evaluable_episodes": test["subjects_with_evaluable_episodes"],
        "partitions_without_evaluable_episodes": zero_event_partitions,
        "zero_event_limitation": "A partition without evaluable events cannot support "
        "event sensitivity or the complete four-way event experiment. "
        "This is a missing denominator, not a formal power threshold.",
        "formal_power_threshold": None,
        "scope": "exploratory hackathon model experiments only",
        "evaluation_event_concentration": {
            p: {
                key: counts[key]
                for key in (
                    "largest_subject_ids",
                    "largest_subject_evaluable_episodes",
                    "largest_subject_evaluable_episode_fraction",
                )
            }
            for p, counts in statistics["partitions"].items()
            if p != "training"
        },
    }
    subject_support = [
        {"partition": partition, "subject_id": subject, "evaluable_episodes": count}
        for partition, counts in statistics["partitions"].items()
        for subject, count in counts["evaluable_episodes_by_subject"].items()
    ]
    pd.DataFrame(subject_support).to_csv(output / "event_support_by_subject.csv", index=False)
    (output / "cohort_report.json").write_text(json.dumps(report, indent=2) + "\n")
    provenance = json.loads((output / "provenance.json").read_text())
    repository = Path(__file__).resolve().parents[2]
    provenance.update(
        {
            "frozen_protocol_sha256": sha256_file(repository / "docs/vitaldb_protocol.md"),
            "acquisition_manifest_sha256": sha256_file(output / "acquisition_manifest.json"),
            "pilot_manifest_sha256": sha256_file(pilot_manifest_path),
            "script_sha256": {
                p.name: sha256_file(p) for p in sorted((repository / "scripts").glob("*.py"))
            },
            "audit_only_track": "SNUADC/ART",
            "waveform_predictors": [],
        }
    )
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    lines = [
        f"# Realized VitalDB cohort: {output.name}",
        "",
        "No model trained or evaluated. "
        "All windows retained; prevalence below is descriptive of this selected prefix.",
        "",
    ]
    keys = [
        "candidate_cases_considered",
        "cases_downloaded_additional",
        "additional_download_bytes",
        "cases_parsed",
        "cases_with_art_mbp",
        "cases_with_eligible_anchors",
        "unique_subjects",
        "unique_eligible_subjects",
        "unique_cases",
        "eligible_windows",
        "positive_windows",
        "negative_windows",
        "positive_rate",
        "confirmed_hypotensive_episodes",
        "evaluable_episodes",
        "episodes_without_forecast_opportunity",
        "unique_positive_subjects",
        "censored_candidate_anchors",
        "monitored_hours",
        "acquisition_stop_reason",
    ]
    lines += [f"- {key}: {report[key]}" for key in keys]
    lines += [
        "",
        "## Realized subject-disjoint split",
        "",
        "| Partition | Subjects | Cases | Windows | Positives | Confirmed | "
        "Evaluable | Prevalence | Positive subjects | Evaluable subjects | "
        "MAP hours | Censored | Negatives |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p, c in statistics["partitions"].items():
        lines.append(
            f"| {p} | {c['subjects']} | {c['cases']} | {c['eligible_windows']} | "
            f"{c['positive_windows']} | {c['confirmed_episodes']} | {c['evaluable_episodes']} | "
            f"{c['positive_window_prevalence']:.4%} | {c['positive_subjects']} | "
            f"{c['subjects_with_evaluable_episodes']} | {c['monitored_hours']:.4f} | "
            f"{c['censored_anchors']} | {c['negative_windows']} |"
        )
    lines += [
        "",
        f"Stratified: {manifest['stratified']}. "
        f"Fallback: {manifest['stratification_fallback_reason']}",
        "",
        "All pilot subjects remain training-only. Manifest covers exactly eligible subjects.",
        "",
    ]
    lines += [
        f"- intersection({key}) = {value}"
        for key, value in statistics["subject_intersections"].items()
    ]
    lines += ["", "### Independent evaluable-event support", ""]
    for partition, counts in statistics["partitions"].items():
        lines.append(
            f"- {partition}: {counts['subjects_with_evaluable_episodes']} independent subjects; "
            f"subject episode counts (including zero): {counts['evaluable_episodes_by_subject']}"
        )
        fraction = counts["largest_subject_evaluable_episode_fraction"]
        share = f"{fraction:.2%}" if fraction is not None else "undefined (zero episodes)"
        lines.append(
            f"  Largest single-subject share: {counts['largest_subject_evaluable_episodes']}/"
            f"{counts['evaluable_episodes']} = {share}; "
            f"subject IDs: {counts['largest_subject_ids']}."
        )
    calibration = statistics["partitions"]["calibration"]
    lines += [
        "",
        "All test evaluable events concentrated in one subject: "
        + str(test["evaluable_episodes"] > 0 and test["subjects_with_evaluable_episodes"] == 1),
        "",
        "Calibration evaluable-event support effectively one subject: "
        + str(calibration["subjects_with_evaluable_episodes"] == 1),
        "",
        "Episode counts are descriptive; no formal statistical-power threshold is imposed.",
    ]
    split_lines = lines[lines.index("## Realized subject-disjoint split") :]
    (output / "realized_split_report.md").write_text("\n".join(split_lines) + "\n")
    lines += [
        "",
        "## Track coverage",
        "",
        "| Signal | Available cases | Missing fraction on surgical grid |",
        "|---|---:|---:|",
    ]
    lines += [
        f"| {s} | {report['track_availability_cases'][s]} | {v['missing_fraction']:.6f} |"
        for s, v in report["track_coverage"].items()
    ]
    lines += [
        "",
        "## Case exclusions",
        "",
        str(report["pilot_case_exclusions"]),
        "",
        "See case_report.csv for every processed case and reason.",
        "",
        "## Subject distributions",
        "",
        "Eligible windows (eligible subjects): "
        + str(report["eligible_windows_per_subject_distribution"]),
        "",
        "Evaluable episodes (eligible subjects, including zero counts): "
        + str(report["evaluable_episodes_per_eligible_subject_distribution"]),
        "",
        "## Human audits",
        "",
        str(report["waveform_audit"]),
        "",
        str(report["boundary_audit"]),
        "",
        "Figures/manifests are in waveform_audit/ and boundary_audit/. "
        "No labels were changed by either audit.",
        "",
        "## Modeling assessment",
        "",
        str(report["modeling_assessment"]),
        "",
        "## Limitations",
        "",
    ]
    lines += [f"- {limitation}" for limitation in report["limitations"]]
    (output / "cohort_report.md").write_text("\n".join(lines) + "\n")
    table = json.loads((output / "table_manifest.json").read_text())
    table["tables"].update(
        {
            str(p.relative_to(output)): sha256_file(p)
            for folder in ("waveform_audit", "boundary_audit")
            for p in sorted((output / folder).rglob("*"))
            if p.is_file()
        }
    )
    table["tables"]["realized_split_manifest.json"] = sha256_file(
        output / "realized_split_manifest.json"
    )
    table["tables"]["acquisition_manifest.json"] = sha256_file(output / "acquisition_manifest.json")
    for name in (
        "cohort_report.json",
        "provenance.json",
        "runtime_manifest.json",
        "feature_schema.json",
        "case_report.csv",
        "event_support_by_subject.csv",
    ):
        table["tables"][name] = sha256_file(output / name)
    (output / "table_manifest.json").write_text(json.dumps(table, indent=2) + "\n")
    integrity = verify_artifacts(output)
    (output / "integrity_report.json").write_text(json.dumps(integrity, indent=2) + "\n")
    return report
