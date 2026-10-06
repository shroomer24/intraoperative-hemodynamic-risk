"""Small, strict TOML configuration with paths anchored to an explicit project root."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PathsConfig:
    raw: Path
    interim: Path
    processed: Path
    artifacts: Path


@dataclass(frozen=True)
class DatasetConfig:
    case_id_column: str | None = None
    subject_id_column: str | None = None
    time_seconds_column: str | None = None
    signal_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class WindowsConfig:
    lookback_seconds: float | None = None
    stride_seconds: float | None = None


@dataclass(frozen=True)
class ProjectConfig:
    paths: PathsConfig
    seed: int = 42
    log_level: str = "INFO"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    windows: WindowsConfig = field(default_factory=WindowsConfig)


def _check_keys(values: dict, allowed: set[str], section: str) -> None:
    extra = values.keys() - allowed
    if extra:
        raise ValueError(f"Unknown keys in {section}: {sorted(extra)}")


def load_config(path: str | Path, *, project_root: str | Path) -> ProjectConfig:
    """Read configuration without creating directories or accessing datasets."""
    with Path(path).open("rb") as handle:
        values = tomllib.load(handle)
    _check_keys(values, {"seed", "log_level", "paths", "dataset", "windows"}, "root")
    seed = values.get("seed", 42)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    level = values.get("log_level", "INFO")
    if not isinstance(level, str) or level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError("log_level must be a standard uppercase logging level")
    sections = {name: values.get(name, {}) for name in ("paths", "dataset", "windows")}
    for name, section in sections.items():
        if not isinstance(section, dict):
            raise ValueError(f"{name} must be a TOML table")
    path_values = sections["paths"]
    _check_keys(path_values, {"raw", "interim", "processed", "artifacts"}, "paths")
    root = Path(project_root).resolve()
    defaults = {
        "raw": "data/raw",
        "interim": "data/interim",
        "processed": "data/processed",
        "artifacts": "artifacts",
    }
    resolved = {}
    for name, default in defaults.items():
        value = path_values.get(name, default)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"paths.{name} must be a nonempty path string")
        resolved[name] = (root / value).resolve()
    dataset = sections["dataset"]
    _check_keys(
        dataset,
        {"case_id_column", "subject_id_column", "time_seconds_column", "signal_columns"},
        "dataset",
    )
    for name in ("case_id_column", "subject_id_column", "time_seconds_column"):
        if name in dataset and (not isinstance(dataset[name], str) or not dataset[name].strip()):
            raise ValueError(f"dataset.{name} must be a nonempty column name")
    signals = dataset.get("signal_columns", [])
    if not isinstance(signals, list) or any(
        not isinstance(s, str) or not s.strip() for s in signals
    ):
        raise ValueError("dataset.signal_columns must be a list of nonempty column names")
    columns = [
        dataset[k]
        for k in ("case_id_column", "subject_id_column", "time_seconds_column")
        if k in dataset
    ]
    if len(set(columns + signals)) != len(columns + signals):
        raise ValueError("Dataset column names must be distinct")
    windows = sections["windows"]
    _check_keys(windows, {"lookback_seconds", "stride_seconds"}, "windows")
    for name, value in windows.items():
        if type(value) not in (int, float) or not 0 < value < float("inf"):
            raise ValueError(f"windows.{name} must be finite and positive")
    return ProjectConfig(
        paths=PathsConfig(**resolved),
        seed=seed,
        log_level=level,
        dataset=DatasetConfig(**{**dataset, "signal_columns": tuple(signals)}),
        windows=WindowsConfig(**windows),
    )
