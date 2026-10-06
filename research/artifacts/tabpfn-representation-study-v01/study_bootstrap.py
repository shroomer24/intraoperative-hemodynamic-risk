"""Use only this repository's namespaced src tree; no external package fallback."""

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "scripts"))
if "intraop" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "intraop", REPO / "src/__init__.py", submodule_search_locations=[str(REPO / "src")]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["intraop"] = module
    spec.loader.exec_module(module)
if Path(sys.modules["intraop"].__file__).resolve() != (REPO / "src/__init__.py").resolve():
    raise RuntimeError("Authoritative repository import differs")
