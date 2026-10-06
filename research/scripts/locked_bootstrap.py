"""Authoritative local repository import, without a package/environment fallback."""

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if "intraop" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "intraop", REPO / "src/__init__.py", submodule_search_locations=[str(REPO / "src")]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["intraop"] = module
    spec.loader.exec_module(module)
if Path(sys.modules["intraop"].__file__).resolve() != REPO / "src/__init__.py":
    raise PermissionError("Authoritative repository import differs")
