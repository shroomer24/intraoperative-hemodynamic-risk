"""Seed helpers and a lightweight, data-free runtime manifest."""

import json
import platform
import random
from dataclasses import asdict
from importlib import metadata
from pathlib import Path

import numpy as np

from intraop.config import ProjectConfig


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python and legacy NumPy RNGs; return an explicit NumPy generator.

    Pass this seed separately to estimators and splitters. PYTHONHASHSEED must be
    set before starting Python. This does not guarantee accelerator determinism.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def write_run_manifest(path: Path, config: ProjectConfig) -> None:
    """Record config and installed packages; never include records or identifiers.

    Dataset fingerprints, code revision, split membership, and label/feature
    schema versions must be added by the eventual training pipeline.
    """
    packages = {}
    for name in (
        "intraop-prediction",
        "numpy",
        "pandas",
        "scikit-learn",
        "vitaldb",
        "wfdb",
        "matplotlib",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    manifest = {"python": platform.python_version(), "packages": packages, "config": asdict(config)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
