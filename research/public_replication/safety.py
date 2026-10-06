"""Explicit owned paths and a public-mode read/write boundary."""

import sys
from pathlib import Path

from public_replication.cohort import RESEARCH, SCOPE

REFERENCES = {"feature_schema.json", "feature_sets.json", "selected_models.json"}


def owned_output(path: Path, *, existing=False) -> Path:
    """Only direct children of research/outputs/public-replication* are allowed."""
    path = path.absolute()
    outputs = RESEARCH / "outputs"
    if path.parent != outputs or not path.name.startswith("public-replication"):
        raise PermissionError(
            "Public mode requires a separate research/outputs/public-replication path"
        )
    for ancestor in [path, *path.parents]:
        if ancestor.is_symlink():
            raise PermissionError("Symlinked output or ancestor forbidden")
    if path.resolve().parent != outputs.resolve():
        raise PermissionError("Public output escapes its declared root")
    if path.exists() != existing:
        raise FileExistsError("Output existence differs; never overwrite or resume an attempt")
    return path


def install_io_boundary(output: Path, reuse: Path | None = None) -> None:
    """Refuse local reads outside code/dependencies and explicit public inputs.

    This is an accidental-dependency guard, not a sandbox for hostile Python.
    No private source discovery or protected execution loader is authorized.
    """
    code = RESEARCH.resolve()
    dependencies = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
    roots = {*dependencies, output.resolve()}
    if reuse is not None:
        roots.add(reuse.resolve())

    def hook(event, args):
        if event != "open" or isinstance(args[0], int):
            return
        path = Path(args[0]).resolve()
        mode, flags = args[1:3]
        writing = bool(flags & 3) if mode is None else any(c in mode for c in "wax+")
        if writing:
            if not path.is_relative_to(output):
                raise PermissionError("Public mode write outside its new output forbidden")
            return
        if reuse and path == reuse.parent.parent / "scope.json":
            return  # identity marker only; no prior tables, predictions or outcomes
        if any(path.is_relative_to(root) for root in dependencies):
            return  # includes a documented virtualenv nested inside the checkout
        if path.is_relative_to(code):
            rel = path.relative_to(code)
            if not rel.parts:
                raise PermissionError("Undeclared root directory open forbidden")
            if path.is_relative_to(output) or (reuse and path.is_relative_to(reuse)):
                return
            if rel.parts[0] in {"src", "scripts", "public_replication"} and path.suffix == ".py":
                return
            if rel == Path("public_replication/public_cohort_manifest.json"):
                return
            if rel.parts[0] == "reference" and path.name in REFERENCES:
                return
            raise PermissionError("Private, historical or sealed research input forbidden")
        if any(path.is_relative_to(root) for root in roots):
            return
        # Certificate/runtime files are not scientific input; no home-directory fallback.
        if any(
            path.is_relative_to(Path(root))
            for root in ["/etc/ssl", "/proc", "/sys", "/usr/share/zoneinfo"]
        ) or str(path) in {"/dev/null", "/dev/urandom", "/etc/localtime"}:
            return
        raise PermissionError("Undeclared filesystem dependency forbidden in public mode")

    sys.addaudithook(hook)


def check_scope(output: Path) -> None:
    import json

    if json.loads((output / "scope.json").read_text()) != {"scope": SCOPE}:
        raise PermissionError("Public output scope missing or different")
