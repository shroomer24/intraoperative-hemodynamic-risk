"""Small prespecified model matrix; model inputs contain predictors only."""

import contextlib
import io
import logging

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class PrevalenceClassifier:
    classes_ = np.array([0, 1])

    def fit(self, X, y):
        self.prevalence_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        return np.tile([1 - self.prevalence_, self.prevalence_], (len(X), 1))


def build_estimator(kind: str, config: dict):
    if kind == "prevalence":
        return PrevalenceClassifier()
    if kind == "logistic":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(**config)),
            ]
        )
    if kind == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(**config)
    if kind == "tabpfn":
        from tabpfn_client import TabPFNClassifier

        if config["model_path"] != "v3.5_default":
            raise ValueError("TabPFN model substitution forbidden")
        return TabPFNClassifier(**config)
    raise ValueError("Unknown probability estimator kind")


@contextlib.contextmanager
def quiet_hosted_call():
    """SDK chatter stays in memory; caller retains sanitized failures separately."""
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        logging.disable(previous)


def positive_probabilities(model, X: np.ndarray, *, require_ordered_classes=False) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    classes = np.asarray(model.classes_)
    if classes.ndim != 1 or len(classes) != 2 or set(classes.tolist()) != {0, 1}:
        raise ValueError("Model must expose exactly classes 0 and 1")
    if require_ordered_classes and classes.tolist() != [0, 1]:
        raise ValueError("TabPFN classes must be [0, 1]")
    if probabilities.shape != (len(X), 2):
        raise ValueError("Probability row count or binary shape differs")
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    ):
        raise ValueError("Invalid probabilities")
    column = int(np.flatnonzero(classes == 1)[0])
    return probabilities[:, column]
