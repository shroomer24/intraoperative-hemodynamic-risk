"""Fixed TEST diagnostics and shared subject bootstrap; no fitting or interpretation."""

import os

import final_test_support as s
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

from intraop.evaluation.calibration import reliability
from intraop.evaluation.development import development_metrics


def apply_mapping(p, mapping):
    clipped = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return expit(mapping["coefficient"] * np.log(clipped / (1 - clipped)) + mapping["intercept"])


def subject_bootstrap(y, groups, scores, *, replicates=1000, seed=42):
    groups = np.asarray(groups)
    subjects, codes = np.unique(groups, return_inverse=True)
    positions = [np.flatnonzero(codes == i) for i in range(len(subjects))]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(subjects), size=(replicates, len(subjects)))
    values = {name: {"average_precision": [], "auroc": []} for name in scores}
    defined = []
    for draw in draws:
        index = np.concatenate([positions[i] for i in draw])
        target = np.asarray(y)[index]
        valid = np.unique(target).size == 2
        defined.append(valid)
        if not valid:
            continue
        # This one index is shared across every model and raw/calibrated output.
        for name, score in scores.items():
            prediction = np.asarray(score)[index]
            values[name]["average_precision"].append(
                float(average_precision_score(target, prediction))
            )
            values[name]["auroc"].append(float(roc_auc_score(target, prediction)))
    intervals = {
        name: {
            metric: {
                "valid_replicates": len(v),
                "undefined_replicates": replicates - len(v),
                "percentile_95": np.percentile(v, [2.5, 97.5]).tolist() if v else None,
            }
            for metric, v in row.items()
        }
        for name, row in values.items()
    }
    return intervals, draws, codes, np.asarray(defined, dtype=bool)


def diagnostics(raw, y, groups, mappings):
    if tuple(raw) != s.MODELS:
        raise ValueError("All seven fixed models required in locked order")
    s.verify_mappings(mappings)
    definition_before = s.a.digest(mappings)
    calibrated, metrics, tables, bootstrap_scores = {}, {}, {}, {}
    for name in s.MODELS:
        probability = name != "current_map"
        p = raw[name]
        result = development_metrics(y, p, probability_output=probability)
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
        metrics[name + "/raw"] = result
        bootstrap_scores[name + "/raw"] = p
        if probability:
            tables[name + "/raw"] = reliability(y, p)
        if name in s.LEARNED:
            q = apply_mapping(p, mappings[name])
            if not np.isfinite(q).all() or (q < 0).any() or (q > 1).any():
                raise ValueError("Calibrated probabilities must be finite and bounded")
            calibrated[name] = q
            metrics[name + "/calibrated"] = development_metrics(y, q, probability_output=True)
            tables[name + "/calibrated"] = reliability(y, q)
            bootstrap_scores[name + "/calibrated"] = q
    intervals, draws, codes, defined = subject_bootstrap(
        y, groups, bootstrap_scores, replicates=1000, seed=42
    )
    if s.a.digest(mappings) != definition_before:
        raise PermissionError("Metrics cannot alter calibration mappings")
    return calibrated, metrics, tables, intervals, draws, codes, defined


def plot(path, tables):
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
    s.a.atomic_binary(path, lambda stream: figure.savefig(stream, format="png", dpi=160))
    plt.close(figure)


def publish(query, raw, mappings, identities, registry, prep, ph):
    y = query.y.to_numpy(dtype=int)
    before = s.a.digest(registry)
    calibrated, metrics, tables, intervals, draws, codes, defined = diagnostics(
        raw, y, query.groups, mappings
    )
    if s.a.digest(registry) != before:
        raise PermissionError("TEST diagnostics cannot change configurations")
    staged = s.ROOT / "publication"
    staged.mkdir(exist_ok=False)
    s.a.copy_immutable(s.REGISTRY, staged / "source_registry.json")
    s.a.copy_immutable(s.REGISTRY.with_suffix(".sha256"), staged / "source_registry.sha256")
    for name in s.MODELS:
        arrays = {"query_position": np.arange(len(y)), "true_label": y}
        arrays["raw_score" if name == "current_map" else "raw_probability"] = raw[name]
        if name in calibrated:
            arrays["calibrated_probability"] = calibrated[name]
        s.a.atomic_binary(
            staged / "predictions" / f"{name}.npz",
            lambda stream, arr=arrays: np.savez(stream, **arr),
        )
        frame = pd.DataFrame(
            {
                "query_position": np.arange(len(y)),
                "true_label": y,
                "raw_probability": raw[name] if name != "current_map" else np.nan,
                "raw_score": raw[name] if name == "current_map" else np.nan,
                "calibrated_probability": calibrated.get(name, np.full(len(y), np.nan)),
                "model": name,
                "split": "test",
            }
        )
        text = frame.to_csv(index=False, float_format="%.17g", na_rep="")
        s.a.atomic_binary(
            staged / "predictions" / f"{name}.csv",
            lambda stream, value=text: stream.write(value.encode()),
        )
    s.a.atomic_binary(
        staged / "bootstrap_draws.npz",
        lambda stream: np.savez(
            stream,
            subject_ordinal_draws=draws,
            query_subject_ordinal=codes,
            replicate_defined=defined,
        ),
    )
    records = {
        "metrics.json": metrics,
        "reliability.json": tables,
        "subject_bootstrap.json": {
            "protocol": s.TEST_PROTOCOL,
            "intervals": intervals,
            "draws_sha256": s.a.file_hash(staged / "bootstrap_draws.npz"),
            "shared_draws_all_models": True,
            "subject_identifiers_published": False,
        },
        "alignment.json": {
            "rows": len(y),
            "subjects": query.groups.nunique(),
            "positive_labels": int(y.sum()),
            "query_source_order_sha256": s.a.row_hash(query.X.index),
            "ordinal_position_sha256": s.a.row_hash(np.arange(len(y))),
            "labels_sha256": s.a.array_hash(y),
            "group_ordinal_sha256": s.a.row_hash(codes),
            "patient_identifiers_published": False,
            "source_registry_sha256": s.REGISTRY_HASH,
            "model_lock_sha256": s.a.LOCK_HASH,
        },
        "model_identities.json": identities,
        "environment_provenance.json": {
            "python_executable": s.a.PYTHON,
            "python_version": prep["python_version"],
            "package_versions": prep["package_versions"],
            "execution_preparation_sha256": ph,
            "model_lock_sha256": s.a.LOCK_HASH,
            "calibration_receipt_sha256": s.CALIBRATION_RECEIPT_HASH,
            "calibration_parameters_sha256": s.MAPPINGS_HASH,
            "requested_TabPFN_constructor": s.a.CONSTRUCTOR,
            "interpretation_embedded": False,
            "Platt_refit_calls": 0,
            "training_context": "frozen TRAINING only",
        },
    }
    for name, value in records.items():
        s.a.seal_json(staged / name, value)
    os.environ["MPLCONFIGDIR"] = str(s.ROOT / "plot_config")
    plot(staged / "reliability.png", tables)
    return staged
