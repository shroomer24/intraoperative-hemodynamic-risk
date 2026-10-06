"""Run after editable installation: python scripts/check_environment.py."""

import sys
from importlib import metadata

from intraop import __version__


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"intraop: {__version__}")
    for package in ("numpy", "pandas", "scikit-learn"):
        print(f"{package}: {metadata.version(package)}")
    print("Environment imports succeeded; no data was read and no model was trained.")


if __name__ == "__main__":
    main()
