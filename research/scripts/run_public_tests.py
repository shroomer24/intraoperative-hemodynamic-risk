"""Public synthetic-only suite; no acquisition or hosted inference."""

from locked_bootstrap import REPO  # isort: skip
import os
import unittest

os.chdir(REPO)
result = unittest.TextTestRunner(verbosity=1).run(
    unittest.defaultTestLoader.discover("tests", top_level_dir=str(REPO))
)
raise SystemExit(not result.wasSuccessful())
