"""Public replication cohort — not the sealed evaluation cohort."""

from locked_bootstrap import REPO  # isort: skip
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO))
from public_replication.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
