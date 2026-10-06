"""Exactly five prespecified Platt fits; immutable raw sources, no hosted/model reruns."""

import os
import resource
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregation_support as s  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from intraop.evaluation.calibration import apply_platt, fit_platt, reliability  # noqa: E402
from intraop.evaluation.development import development_metrics  # noqa: E402


def plot_reliability(path, tables):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    for name, bins in tables.items():
        occupied = [b for b in bins if b["windows"]]
        axis.plot(
            [b["mean_probability"] for b in occupied],
            [b["observed_frequency"] for b in occupied],
            marker="o",
            label=name,
        )
    axis.set(
        xlabel="Mean predicted probability",
        ylabel="Observed positive fraction",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.legend(fontsize=7)
    figure.tight_layout()
    s.atomic_binary(path, lambda stream: figure.savefig(stream, format="png", dpi=160))
    plt.close(figure)


def compute(raw, y, *, fit=fit_platt, apply=apply_platt, metrics=development_metrics):
    """Fits precede diagnostics; no metric can choose a model or calibration."""
    if set(raw) != set(s.MODEL_NAMES) or np.unique(y).tolist() != [0, 1]:
        raise ValueError("All seven raw models and both calibration label classes required")
    parameters = {}
    calibrated = {}
    for name in s.CALIBRATED_MODELS:
        parameters[name] = fit(raw[name], y)
        parameters[name]["fitting_parameters"] = dict(s.CALIBRATION_PROTOCOL["parameters"])
        calibrated[name] = apply(raw[name], parameters[name])
        p = calibrated[name]
        if p.shape != y.shape or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError("Calibrated probability contract differs")
    summaries, tables = {}, {}
    for name in s.MODEL_NAMES:
        probability = name != "current_map"
        result = metrics(y, raw[name], probability_output=probability)
        if not probability:
            result = {
                k: result[k]
                for k in [
                    "windows",
                    "positives",
                    "average_precision",
                    "auroc",
                    "probability_output",
                ]
            }
        summaries[name + "/raw"] = result
        if probability:
            tables[name + "/raw"] = reliability(y, raw[name])
        if name in s.CALIBRATED_MODELS:
            summaries[name + "/calibrated"] = metrics(y, calibrated[name], probability_output=True)
            tables[name + "/calibrated"] = reliability(y, calibrated[name])
    return parameters, calibrated, summaries, tables


def execute():
    loader = s.CalibrationLabels()
    loader.authorize()
    s.atomic_json(s.ATTEMPT / "worker_claim.json", {"status": "CLAIMED"})
    sequence = 0
    stage = None

    def mark(value):
        nonlocal sequence, stage
        sequence += 1
        s.atomic_json(
            s.ATTEMPT / "stages" / f"{sequence:02d}_{value}.json",
            {"stage": value, "timestamp_utc": s.stamp()},
        )
        stage = value

    try:
        mark("WORKER_STARTED")
        raw = s.raw_outputs(loader.registry)
        y, index = loader.load("calibration")
        mark("SOURCES_AND_CALIBRATION_LABELS_VALIDATED")
        frozen_definition = s.digest(loader.lock["models"])
        if not np.allclose(
            raw["prevalence"], loader.lock["training_context"]["prevalence"], atol=1e-15, rtol=0
        ):
            raise ValueError("Fixed training prevalence differs")
        mark("FIVE_PRESPECIFIED_PLATT_FITS_STARTED")
        mappings, calibrated, metrics, tables = compute(raw, y)
        mark("FIVE_PRESPECIFIED_PLATT_FITS_AND_DIAGNOSTICS_COMPLETED")
        if s.digest(loader.lock["models"]) != frozen_definition:
            raise PermissionError("Diagnostics cannot change scientific configuration")
        loader.authorize()
        publication = s.ATTEMPT / "publication"
        publication.mkdir(exist_ok=False)
        s.copy_immutable(s.REGISTRY, publication / "source_registry.json")
        s.copy_immutable(s.REGISTRY.with_suffix(".sha256"), publication / "source_registry.sha256")
        s.seal_json(publication / "calibration_parameters.json", mappings)
        alignment = {
            "partition": "calibration",
            "rows": len(y),
            "subjects": s.SUBJECTS,
            "positive_labels": int(y.sum()),
            "labels_array_sha256": s.array_hash(y),
            "query_source_order_sha256": s.row_hash(index),
            "published_row_key": "query_position_ordinal_0_based_not_patient_identifier",
            "ordinal_position_sha256": s.row_hash(np.arange(len(y))),
            "source_registry_sha256": s.REGISTRY_HASH,
            "model_lock_sha256": s.LOCK_HASH,
            "calibration_protocol_sha256": s.PROTOCOL_HASH,
            "label_table_sha256": loader.lock["table_hashes"]["labels.csv"],
            "metadata_routing_table_sha256": loader.lock["table_hashes"]["metadata.csv"],
            "split_manifest_sha256": loader.lock["split_manifest_hash"],
            "raw_model_configuration_digest": frozen_definition,
            "patient_identifiers_published": False,
            "calibration_metrics_diagnostic_only": True,
        }
        s.atomic_binary(
            publication / "calibration_labels.npy",
            lambda stream: np.save(stream, y, allow_pickle=False),
        )
        alignment["label_file_sha256"] = s.file_hash(publication / "calibration_labels.npy")
        s.seal_json(publication / "calibration_alignment.json", alignment)
        for name in s.MODEL_NAMES:
            frame = pd.DataFrame({"query_position": np.arange(len(y)), "true_label": y})
            frame["raw_probability"] = raw[name] if name != "current_map" else np.nan
            frame["raw_score"] = raw[name] if name == "current_map" else np.nan
            frame["calibrated_probability"] = calibrated.get(name, np.full(len(y), np.nan))
            frame["model"] = name
            frame["split"] = "calibration"
            csv_text = frame.to_csv(index=False, float_format="%.17g", na_rep="")
            s.atomic_binary(
                publication / "predictions" / f"{name}.csv",
                lambda stream, text=csv_text: stream.write(text.encode()),
            )
            arrays = {"query_position": np.arange(len(y)), "true_label": y}
            arrays["raw_score" if name == "current_map" else "raw_probability"] = raw[name]
            if name in calibrated:
                arrays["calibrated_probability"] = calibrated[name]
            s.atomic_binary(
                publication / "predictions" / f"{name}.npz",
                lambda stream, values=arrays: np.savez(stream, **values),
            )
        s.seal_json(publication / "metrics.json", metrics)
        s.seal_json(publication / "reliability.json", tables)
        os.environ["MPLCONFIGDIR"] = str(s.ATTEMPT / "plot_config")
        plot_reliability(publication / "reliability.png", tables)
        mark("OUTPUTS_STAGED")
        loader.authorize()
        s.seal_json(
            s.ATTEMPT / "worker_completion.json",
            {
                "status": "STAGED_COMPLETE",
                "scope": "FINAL_CALIBRATION_AGGREGATION",
                "execution_preparation_sha256": loader.preparation_hash,
                "source_registry_sha256": s.REGISTRY_HASH,
                "partition_access": loader.access_log,
                "Platt_models": list(s.CALIBRATED_MODELS),
                "Platt_fit_calls": 5,
                "raw_model_fit_or_predict_calls": 0,
                "hosted_calls": 0,
                "TEST_accessed": False,
                "files_sha256": s.receipt_files(publication),
                "automatic_retry": False,
                "timestamp_utc": s.stamp(),
            },
        )
        return 0
    except BaseException as exc:
        s.seal_json(
            s.ATTEMPT / "failure.json",
            {
                "status": "FAILED",
                "stage": stage,
                "exception_class": s.safe_exception(exc),
                "raw_exception_messages_retained": False,
                "automatic_retry": False,
                "timestamp_utc": s.stamp(),
            },
        )
        return 2


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    s.install_execution_firewall()
    if len(sys.argv) != 1:
        return 2
    try:
        return execute()
    except BaseException:
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
