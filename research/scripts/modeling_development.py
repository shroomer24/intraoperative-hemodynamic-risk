"""Phase 1 only. Hosted execution is for the owner's normal terminal."""

import argparse
import importlib.metadata
import json
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from intraop.data.modeling import (
    DevelopmentPartitions,
    predictor_array,
    read_json,
    sha256,
    write_json,
)
from intraop.evaluation.development import (
    development_metrics,
    export_predictions,
    select_candidate,
)
from intraop.evaluation.secure_errors import sanitized_traceback
from intraop.models.benchmarks import (
    build_estimator,
    positive_probabilities,
    quiet_hosted_call,
)


def validate_environment(plan: dict, *, require_token: bool) -> dict:
    import os

    if sys.executable != plan["python_executable"]:
        raise RuntimeError("Python executable differs from the prepared plan")
    if sys.version.split()[0] != plan["python_version"]:
        raise RuntimeError("Python version differs from the prepared plan")
    versions = {name: importlib.metadata.version(name) for name in plan["package_versions"]}
    if versions != plan["package_versions"] or versions["tabpfn-client"] != "0.6.1":
        raise RuntimeError("Package version differs from the prepared plan")
    present = bool(os.environ.get("TABPFN_TOKEN"))
    if require_token and not present:
        raise RuntimeError("TABPFN_TOKEN is unavailable; value must never be printed")
    return {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "packages": versions,
        "TABPFN_TOKEN_present": present,
        "tabpfn_model_identifier": "v3.5_default",
        "hosted_execution_boundary": "normal_terminal",
        "credential_values_recorded": False,
    }


def validate_inputs(repo: Path, plan: dict) -> tuple:
    if plan["phase"] != "development_only" or plan["allowed_partitions"] != ["training", "tuning"]:
        raise PermissionError("This entrypoint permits development only")
    if plan["selection_metric"] != "tuning_average_precision":
        raise ValueError("Selection metric changed")
    if plan["tabpfn_model_identifier"] != "v3.5_default":
        raise ValueError("TabPFN model substitution forbidden")
    if (repo / "artifacts/modeling-v01/model_lock_manifest.json").exists():
        raise PermissionError("Development is forbidden once a model lock exists")
    import intraop

    if Path(intraop.__file__).resolve() != (repo / "src/__init__.py").resolve():
        raise RuntimeError("Imported repository package differs from the authoritative checkout")
    for name, expected in plan["code_hashes"].items():
        if sha256(repo / name) != expected:
            raise ValueError(f"Prepared code differs: {name}")
    if (
        sha256(repo / "artifacts/modeling-v01/feature_sets.json")
        != plan["feature_sets_file_sha256"]
    ):
        raise ValueError("Frozen feature sets file differs")
    loader = DevelopmentPartitions(
        repo / plan["cohort_checkpoint"],
        repo / "artifacts/modeling-v01",
        plan["dataset_manifest_hashes"],
    )
    if loader.feature_sets["full"] != plan["feature_sets"]["full"]:
        raise ValueError("Prepared full feature contract differs")
    if loader.feature_sets["map_only"] != plan["feature_sets"]["map_only"]:
        raise ValueError("Prepared MAP-only feature contract differs")
    training = loader.load("training")
    tuning = loader.load("tuning")
    if set(training.groups) & set(tuning.groups):
        raise ValueError("Development subjects overlap")
    if len(training.X) != 13975 or len(tuning.X) != 3193:
        raise ValueError("Frozen development window counts differ")
    for data in (training, tuning):
        if np.unique(data.y).tolist() != [0, 1]:
            raise ValueError("Frozen development partition lacks a label class")
        if not data.metadata.window_id.is_unique:
            raise ValueError("Prediction window identities are not unique")
    return loader, training, tuning


def external_smoke(training, names: list[str]) -> dict:
    """Same 256 TRAINING rows as the prior smoke; never changes later training."""
    rng = np.random.default_rng(42)
    positions = np.sort(rng.choice(len(training.X), 256, replace=False))
    remaining = np.setdiff1d(np.arange(len(training.X)), positions)
    query_positions = rng.choice(remaining, 12, replace=False)
    query_positions = np.r_[query_positions, query_positions[:4]]
    fit = training.subset(positions)
    if np.unique(fit.y).tolist() != [0, 1]:
        raise ValueError("The exact prior smoke subset lacks both classes")
    Xfit = predictor_array(fit, names)
    Xquery = predictor_array(training, names)[query_positions]
    with quiet_hosted_call():
        model = build_estimator("tabpfn", {"model_path": "v3.5_default", "random_state": 42})
        model.fit(Xfit, fit.y.to_numpy(dtype=int))
        probabilities = positive_probabilities(model, Xquery, require_ordered_classes=True)
        reverse = positive_probabilities(model, Xquery[::-1].copy(), require_ordered_classes=True)
    if not np.allclose(probabilities, reverse[::-1], atol=1e-5):
        raise ValueError("Smoke reversed-row order check failed")
    if not np.allclose(probabilities[:4], probabilities[12:], atol=1e-5):
        raise ValueError("Smoke duplicate-row order check failed")
    return {
        "status": "PASS",
        "partition": "training",
        "fit_rows": 256,
        "fit_class_counts": {"0": int((fit.y == 0).sum()), "1": int((fit.y == 1).sum())},
        "predictor_count": 74,
        "query_rows": 16,
        "binary_probability_shape": [16, 2],
        "classes": [0, 1],
        "positive_probability_column": 1,
        "positive_class_semantics": (
            "1 = new sustained hypotension episode beginning within 5 minutes"
        ),
        "row_order_verified_by_reversal_and_duplicates": True,
        "model_identifier": "v3.5_default",
        "random_state": 42,
        "uploaded_fields": ["numeric frozen predictor matrix", "binary TRAINING labels"],
    }


def run_development(repo: Path, plan_path: Path, expected_plan_hash: str) -> dict:
    if sha256(plan_path) != expected_plan_hash:
        raise ValueError("Development plan SHA-256 mismatch")
    plan = read_json(plan_path)
    environment = validate_environment(plan, require_token=True)
    loader, training, tuning = validate_inputs(repo, plan)
    output = repo / "artifacts/modeling-v01/development"
    output.mkdir(exist_ok=False)
    started = datetime.now(UTC).isoformat()
    run_id = "modeling-v01-development-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    write_json(
        output / "started_manifest.json",
        {
            "status": "STARTED",
            "run_id": run_id,
            "timestamp_utc": started,
            "phase": "development_only",
            "development_plan_sha256": expected_plan_hash,
            "environment": environment,
            "partition_access_log": loader.access_log,
        },
        exclusive=True,
    )
    stage = "external_training_smoke"
    try:
        print("Running external TRAINING-only access smoke", flush=True)
        smoke = external_smoke(training, loader.features)
        write_json(output / "external_training_smoke.json", smoke, exclusive=True)
        records, selected, selected_predictions = [], {}, []
        ytrain = training.y.to_numpy(dtype=int)
        candidates_dir = output / "candidates"
        candidates_dir.mkdir()
        models_dir = output / "models"
        models_dir.mkdir()
        for model_name, definition in plan["models"].items():
            names = definition["features"]
            Xtrain = predictor_array(training, names)
            Xtuning = predictor_array(tuning, names)
            local_records, estimators, scores_by_candidate = [], {}, {}
            for candidate in definition["candidates"]:
                candidate_id = candidate["candidate_id"]
                stage = f"{model_name}/{candidate_id}"
                print(f"Development candidate: {stage}", flush=True)
                probability_output = definition["kind"] != "current_map"
                if definition["kind"] == "current_map":
                    if names != ["map_latest"] or not np.isfinite(Xtuning).all():
                        raise ValueError("Current MAP must be finite in frozen eligible rows")
                    scores = -Xtuning[:, 0]
                    estimator = None
                elif definition["kind"] == "tabpfn":
                    with quiet_hosted_call():
                        estimator = build_estimator("tabpfn", candidate["hyperparameters"])
                        estimator.fit(Xtrain, ytrain)
                        scores = positive_probabilities(
                            estimator,
                            Xtuning,
                            require_ordered_classes=True,
                        )
                else:
                    estimator = build_estimator(definition["kind"], candidate["hyperparameters"])
                    estimator.fit(Xtrain, ytrain)
                    scores = positive_probabilities(estimator, Xtuning)
                metrics = development_metrics(
                    tuning.y, scores, probability_output=probability_output
                )
                record = {
                    "model": model_name,
                    "candidate_id": candidate_id,
                    "simplicity_rank": candidate["simplicity_rank"],
                    "hyperparameters": json.dumps(candidate["hyperparameters"], sort_keys=True),
                    **metrics,
                }
                local_records.append(record)
                records.append(record)
                estimators[candidate_id] = estimator
                scores_by_candidate[candidate_id] = scores
                prediction = export_predictions(
                    tuning,
                    scores,
                    model=model_name,
                    candidate_id=candidate_id,
                    probability_output=probability_output,
                )
                prediction.to_csv(candidates_dir / f"{model_name}__{candidate_id}.csv", index=False)
                # Incremental evidence survives an external-service interruption.
                pd.DataFrame(records).to_csv(output / "candidate_metrics.csv", index=False)
            winner = select_candidate(local_records)
            winner_id = winner["candidate_id"]
            config = next(c for c in definition["candidates"] if c["candidate_id"] == winner_id)
            estimator = estimators[winner_id]
            classes = np.asarray(estimator.classes_).tolist() if estimator is not None else None
            selected[model_name] = {
                "kind": definition["kind"],
                "features": names,
                "preprocessing": definition["preprocessing"],
                "selected_configuration": config,
                "tuning_metrics": winner,
                "selection_metric": plan["selection_metric"],
                "tie_break": plan["tie_break"],
                "fit_partition": "training",
                "selection_partition": "tuning",
                "training_rows": len(training.X),
                "training_subjects": int(training.groups.nunique()),
                "training_prevalence": float(training.y.mean()),
                "positive_class": 1,
                "classes": classes,
                "probability_column": classes.index(1) if classes is not None else None,
            }
            if definition["kind"] == "tabpfn":
                # Do not pickle a client, auth state, caches, or remote internals.
                write_json(
                    models_dir / f"{model_name}.json",
                    {
                        "model_identifier": "v3.5_default",
                        "tabpfn_client_version": "0.6.1",
                        "constructor_arguments": config["hyperparameters"],
                        "classes": [0, 1],
                        "fitted_train_rows": len(training.X),
                        "hosted_inference": True,
                        "training_labels_from": "training_only",
                        "input_columns": names,
                        "preprocessing": definition["preprocessing"],
                        "remote_internals_serialized": False,
                    },
                    exclusive=True,
                )
            elif estimator is not None:
                joblib.dump(estimator, models_dir / f"{model_name}.joblib")
            selected_predictions.append(
                export_predictions(
                    tuning,
                    scores_by_candidate[winner_id],
                    model=model_name,
                    candidate_id=winner_id,
                    probability_output=probability_output,
                )
            )
            write_json(output / "selected_configs.json", selected)
        pd.DataFrame(
            [
                select_candidate([r for r in records if r["model"] == name])
                for name in plan["models"]
            ]
        ).to_csv(output / "tuning_metrics.csv", index=False)
        pd.concat(selected_predictions, ignore_index=True).to_csv(
            output / "tuning_predictions.csv",
            index=False,
        )
        # Verify immutable inputs and source hashes again before declaring completion.
        validate_inputs(repo, plan)
        files = {
            str(path.relative_to(output)): sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        }
        completion = {
            "status": "COMPLETE",
            "phase": "development_only",
            "run_id": run_id,
            "started_utc": started,
            "completed_utc": datetime.now(UTC).isoformat(),
            "development_plan_sha256": expected_plan_hash,
            "environment": environment,
            "dataset_manifest_hashes": plan["dataset_manifest_hashes"],
            "split_manifest_hash": plan["dataset_manifest_hashes"]["realized_split_manifest.json"],
            "feature_sets": plan["feature_sets"],
            "code_hashes": plan["code_hashes"],
            "models_completed": list(selected),
            "candidate_count": len(records),
            "selected_configurations_file": "selected_configs.json",
            "selection_metric": plan["selection_metric"],
            "tie_break": plan["tie_break"],
            "training_context": plan["training_context"],
            "smoke_test": smoke,
            "partition_access_log": loader.access_log,
            "calibration_accessed": False,
            "test_values_parsed": False,
            "test_evaluation_count": 0,
            "model_lock_written": False,
            "row_alignment": (
                "source order; numeric query arrays; row-count checks; positional export"
            ),
            "prediction_rows_per_selected_model": len(tuning.X),
            "files_sha256": files,
            "credential_values_recorded": False,
            "review_package": "development_review.zip",
        }
        write_json(output / "development_completion_manifest.json", completion, exclusive=True)
        stage = "aggregate_artifact_packaging"
        # Explicit allowlist: no patient predictions, fitted binaries, raw data,
        # credential files, client state, or environment-file contents.
        review_files = [
            "development_completion_manifest.json",
            "started_manifest.json",
            "external_training_smoke.json",
            "candidate_metrics.csv",
            "tuning_metrics.csv",
            "selected_configs.json",
            "models/tabpfn_map.json",
            "models/tabpfn_full.json",
        ]
        with zipfile.ZipFile(
            output / "development_review.zip", "x", zipfile.ZIP_DEFLATED
        ) as archive:
            for name in review_files:
                archive.write(output / name, name)
            archive.write(plan_path, "development_plan.json")
            archive.writestr("development_plan.sha256", expected_plan_hash + "\n")
        print(
            "Development COMPLETE. Stop here; no model lock/calibration/TEST execution.", flush=True
        )
        print(
            "Review artifact: artifacts/modeling-v01/development/development_review.zip", flush=True
        )
        return completion
    except Exception as exc:
        trace = sanitized_traceback(exc)
        (output / "failure_traceback.txt").write_text(trace)
        write_json(
            output / "failure_manifest.json",
            {
                "status": "FAILED",
                "run_id": run_id,
                "failed_stage": stage,
                "exception_type": type(exc).__name__,
                "sanitized_traceback": "failure_traceback.txt",
                "partition_access_log": loader.access_log,
                "test_evaluation_count": 0,
                "model_lock_written": False,
                "credential_values_recorded": False,
            },
            exclusive=True,
        )
        print(trace, flush=True)
        raise SystemExit(2) from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true", help="No fitting or hosted calls")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    plan_path = repo / "artifacts/modeling-v01/development_plan.json"
    expected = (plan_path.with_suffix(".sha256")).read_text().strip()
    try:
        if sha256(plan_path) != expected:
            raise ValueError("Development plan SHA-256 mismatch")
        if args.validate_only:
            plan = read_json(plan_path)
            environment = validate_environment(plan, require_token=False)
            loader, training, tuning = validate_inputs(repo, plan)
            print(
                json.dumps(
                    {
                        "status": "VALIDATION_ONLY_PASS",
                        "fit_calls": 0,
                        "hosted_calls": 0,
                        "partition_access_log": loader.access_log,
                        "training_shape": list(training.X.shape),
                        "tuning_shape": list(tuning.X.shape),
                        "full_feature_count": len(loader.features),
                        "map_feature_count": len(loader.feature_sets["map_only"]),
                        "environment": environment,
                    },
                    indent=2,
                ),
                flush=True,
            )
        else:
            run_development(repo, plan_path, expected)
    except Exception as exc:
        print(sanitized_traceback(exc), flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
