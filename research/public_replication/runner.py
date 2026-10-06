"""New public execution capability around the original scientific machinery."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from intraop.data.acquisition import METADATA_FILES, acquire_subset, sha256_file
from intraop.data.modeling import predictor_array
from intraop.data.pipeline import build_case_tables
from intraop.data.vitaldb_reader import load_clinical_metadata
from intraop.evaluation.subject_splits import make_subject_split, partition_positions
from intraop.features.core import FEATURE_NAMES
from public_replication.cohort import (
    MAX_BYTES,
    RESEARCH,
    SCOPE,
    SEED,
    acquire_metadata,
    acquire_recordings,
    canonical_bytes,
    validate_manifest,
)
from public_replication.safety import check_scope, install_io_boundary, owned_output

MODELS = (
    "prevalence",
    "current_map",
    "logistic_map",
    "logistic_full",
    "xgboost",
    "tabpfn_map",
    "tabpfn_full",
)


def save(path: Path, value: dict) -> None:
    with path.open("xb") as stream:
        stream.write(canonical_bytes(value))


def copy_public_source(source: Path, destination: Path) -> None:
    """Explicit previously generated public input cache; no hidden cache lookup."""
    previous = owned_output(source.parent.parent, existing=True)
    if source != previous / "source/vitaldb-1.0.0":
        raise PermissionError("Only a prior public replication input directory may be reused")
    check_scope(previous)
    for p in [source, *source.parents]:
        if p.is_symlink():
            raise PermissionError("Symlinked input forbidden")
    manifest = validate_manifest(source)
    names = list(METADATA_FILES) + [
        f"vital_files/{c:04d}.vital"
        for c in manifest["public_vitaldb_case_ids_in_acquisition_order"]
    ]
    for name in names:
        old = source / name
        if old.is_symlink():
            raise PermissionError("Symlinked input file forbidden")
        if old.exists():
            new = destination / name
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(old, new)


def build_public_tables(source: Path, output: Path, manifest: dict) -> tuple:
    """The original case builder supplies all scientific labels/features/audits."""
    cases = manifest["public_vitaldb_case_ids_in_acquisition_order"]
    clinical = load_clinical_metadata(source)
    selected = clinical.loc[clinical.caseid.isin(cases)].sort_values("caseid")
    if len(selected) != len(cases):
        raise ValueError("Selected public case absent; no replacement allowed")
    tables = build_case_tables(source, selected)
    data = tables.dataset
    schema = json.loads((RESEARCH / "reference/feature_schema.json").read_text())
    subsets = json.loads((RESEARCH / "reference/feature_sets.json").read_text())
    if data.X.columns.tolist() != schema["feature_names"] or list(FEATURE_NAMES) != subsets["full"]:
        raise ValueError("Original 74-feature schema mismatch")
    if len(subsets["map_only"]) != 18 or subsets["map_only"] != [
        n for n in FEATURE_NAMES if n.startswith("map_")
    ]:
        raise ValueError("Original 18-feature MAP schema mismatch")
    if set(data.X.columns) & set(data.metadata.columns):
        raise ValueError("Metadata entered predictors")
    if np.isinf(data.X.to_numpy()).any():
        raise ValueError("Infinite predictor values")
    output.mkdir()
    data.X.to_csv(output / "public_replication_features.csv", index=False)
    data.X.loc[:, subsets["map_only"]].to_csv(
        output / "public_replication_map_features.csv", index=False
    )
    data.y.to_csv(output / "public_replication_labels.csv", index=False)
    data.metadata.to_csv(output / "public_replication_metadata.csv", index=True)
    tables.episodes.to_csv(output / "public_replication_episodes.csv", index=False)
    tables.anchor_audit.to_csv(output / "public_replication_anchor_audit.csv", index=False)
    split = make_subject_split(sorted(data.groups.unique()), seed=SEED)
    split["scope"] = SCOPE
    split["replication_only"] = True
    save(output / "public_replication_split.json", split)
    positions = partition_positions(data.metadata, split)
    groups = {k: set(data.subset(v).groups) for k, v in positions.items()}
    if any(groups[a] & groups[b] for a in groups for b in groups if a != b):
        raise ValueError("Subject leakage in public replication")
    if any(len(v) == 0 for v in positions.values()):
        raise ValueError("Empty replication partition; no outcome-driven reassignment")
    training = data.subset(positions["training"])
    query_rows = np.sort(np.concatenate([positions[p] for p in positions if p != "training"]))
    query = data.subset(query_rows)
    summary = {
        "scope": SCOPE,
        "requested_cases": len(cases),
        "parsed_cases": sum(r["parsed"] for r in tables.reports),
        "included_cases": sum(r["included"] for r in tables.reports),
        "cases_with_eligible_anchors": data.metadata.case_id.nunique(),
        "excluded_cases": sum(not r["included"] for r in tables.reports),
        "eligible_windows": len(data.y),
        "positive_windows": int(data.y.sum()),
        "confirmed_episodes": len(tables.episodes),
        "evaluable_episodes": int(tables.episodes.evaluable.sum()),
        "censored_anchors": int((tables.anchor_audit.status == "censored").sum()),
        "ineligible_anchors": int((tables.anchor_audit.status == "ineligible").sum()),
        "full_predictors": 74,
        "map_predictors": 18,
        "subject_intersections_empty": True,
        "partition_subject_counts": {"replication_" + k: len(v) for k, v in groups.items()},
        "partition_windows": {"replication_" + k: len(v) for k, v in positions.items()},
        "training_class_counts": {str(k): int(v) for k, v in training.y.value_counts().items()},
        "query_windows": len(query_rows),
        "query_policy": "source-ordered combined non-training replication partitions; "
        "interface verification only; no tuning, calibration fitting or performance estimate",
        "metrics_computed": False,
    }
    np.savez(
        output.parent / "public_model_inputs.npz",
        train_X=predictor_array(training, subsets["full"]),
        train_y=training.y.to_numpy(),
        query_X=predictor_array(query, subsets["full"]),
        query_rows=query_rows,
    )
    return summary, training.y.nunique() == 2


def execute_models(output: Path, binary: bool, *, skip=False, hosted=False, python=None) -> dict:
    result = {}
    model_root = output / "models"
    model_root.mkdir()
    for name in MODELS:
        reason = None
        if skip:
            reason = "model execution disabled explicitly"
        elif name.startswith("tabpfn") and (not hosted or not os.environ.get("TABPFN_TOKEN")):
            reason = "authenticated hosted access required; explicit --with-tabpfn opt-in required"
        elif not binary and name not in {"prevalence", "current_map"}:
            reason = "replication TRAINING has fewer than two classes; no cohort replacement"
        if reason:
            result[name] = {"status": "SKIPPED", "reason": reason, "scope": SCOPE}
            continue
        target = model_root / name
        target.mkdir()
        print(json.dumps({"scope": SCOPE, "stage": "MODEL", "model": name}), flush=True)
        command = [
            python or sys.executable,
            "-B",
            str(RESEARCH / "scripts/public_replication_worker.py"),
            "--run",
            str(output),
            "--model",
            name,
        ]
        with (target / "process.log").open("w") as stream:
            worker = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
        receipt = {
            "scope": SCOPE,
            "exit_code": worker.returncode,
            "signal": -worker.returncode if worker.returncode < 0 else None,
        }
        save(target / "exit.json", receipt)
        if worker.returncode or not (target / "completion.json").exists():
            result[name] = {
                "status": "BLOCKED",
                "reason": "isolated worker did not complete",
                "scope": SCOPE,
                **receipt,
            }
            break  # retain evidence; no automatic retry or later hosted call
        result[name] = json.loads((target / "completion.json").read_text())
    for name in MODELS:
        result.setdefault(
            name, {"status": "BLOCKED", "reason": "prior worker stopped execution", "scope": SCOPE}
        )
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=SCOPE)
    p.add_argument("--output", type=Path, default=RESEARCH / "outputs/public-replication")
    p.add_argument(
        "--reuse-public-source",
        type=Path,
        help="explicit input-only cache from a previous public replication output",
    )
    p.add_argument("--skip-models", action="store_true")
    p.add_argument(
        "--with-tabpfn", action="store_true", help="explicit authenticated hosted opt-in"
    )
    p.add_argument("--model-python", help="explicit conventional/hosted worker interpreter")
    p.add_argument(
        "--selection-only", action="store_true", help="regenerate/verify the public manifest"
    )
    args = p.parse_args(argv)
    output = owned_output(args.output)
    reuse = args.reuse_public_source.absolute() if args.reuse_public_source else None
    if reuse:
        owned_output(reuse.parent.parent, existing=True)
    output.mkdir(parents=True)
    save(output / "scope.json", {"scope": SCOPE})
    install_io_boundary(output, reuse)
    stage = "PUBLIC_METADATA"
    try:
        print(json.dumps({"scope": SCOPE, "stage": stage}), flush=True)
        source = output / "source/vitaldb-1.0.0"
        if reuse:
            copy_public_source(reuse, source)
        acquire_metadata(source)
        manifest = validate_manifest(source)
        save(output / "public_cohort_manifest.json", manifest)
        digest = hashlib.sha256(canonical_bytes(manifest)).hexdigest()
        if args.selection_only:
            save(
                output / "selection_completion.json",
                {"scope": SCOPE, "status": "PASS", "manifest_sha256": digest},
            )
            return 0
        stage = "PUBLIC_RECORDINGS"
        print(json.dumps({"scope": SCOPE, "stage": stage}), flush=True)
        acquire_recordings(source, manifest)
        acquire_subset(
            source,
            manifest["public_vitaldb_case_ids_in_acquisition_order"],
            max_total_bytes=MAX_BYTES,
        )
        stage = "ORIGINAL_CAUSAL_PIPELINE"
        print(json.dumps({"scope": SCOPE, "stage": stage}), flush=True)
        summary, binary = build_public_tables(source, output / "tables", manifest)
        stage = "PUBLIC_MODEL_INTERFACES"
        statuses = execute_models(
            output, binary, skip=args.skip_models, hosted=args.with_tabpfn, python=args.model_python
        )
        hashes = {
            p.relative_to(output).as_posix(): sha256_file(p)
            for p in sorted((output / "tables").iterdir())
            if p.is_file()
        }
        summary.update(
            {
                "selection_manifest_sha256": digest,
                "table_sha256": hashes,
                "models": statuses,
                "sealed_results_replaced": False,
                "original_calibration_used": False,
                "hyperparameter_search": False,
                "status": "BLOCKED"
                if any(s["status"] == "BLOCKED" for s in statuses.values())
                else "PASS",
            }
        )
        save(output / "public_replication_summary.json", summary)
        print(
            json.dumps(
                {
                    "scope": SCOPE,
                    "status": summary["status"],
                    "summary": "public_replication_summary.json",
                }
            ),
            flush=True,
        )
        return 0 if summary["status"] == "PASS" else 1
    except Exception as exc:
        # Never retain arbitrary messages/tracebacks which can contain inputs or credentials.
        save(
            output / "failure.json",
            {
                "scope": SCOPE,
                "status": "BLOCKED",
                "stage": stage,
                "exception_class": type(exc).__name__,
                "automatic_retry": False,
            },
        )
        print(
            json.dumps(
                {
                    "scope": SCOPE,
                    "status": "BLOCKED",
                    "stage": stage,
                    "exception_class": type(exc).__name__,
                }
            ),
            flush=True,
        )
        return 1
