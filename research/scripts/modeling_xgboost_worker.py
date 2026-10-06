"""Fresh exec XGBoost worker. No Torch/TabPFN runtime is permitted."""

import importlib.abc
import sys


class RuntimeBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("torch", "tabpfn", "tabpfn_client"):
            raise RuntimeError("XGBoost process isolation failure: forbidden import attempted")
        return None


if any(name.split(".")[0] in ("torch", "tabpfn", "tabpfn_client") for name in sys.modules):
    raise RuntimeError("XGBoost process isolation failure: runtime contaminated at startup")
sys.meta_path.insert(0, RuntimeBlocker())

from modeling_candidate_worker import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main("xgboost"))
