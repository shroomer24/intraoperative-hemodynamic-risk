import json
import logging
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from intraop.config import load_config
from intraop.logging import configure_logging
from intraop.reproducibility import seed_everything, write_run_manifest


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.config_path = self.root / "default.toml"

    def load(self, text):
        self.config_path.write_text(text, encoding="utf-8")
        return load_config(self.config_path, project_root=self.root)

    def test_defaults_are_infrastructure_only(self):
        config = self.load("seed = 7\n")
        self.assertEqual(config.paths.raw, self.root / "data/raw")
        self.assertIsNone(config.dataset.subject_id_column)
        self.assertIsNone(config.windows.lookback_seconds)
        self.assertFalse(config.paths.raw.exists())

    def test_explicit_schema_and_window_values(self):
        config = self.load(
            '[dataset]\nsubject_id_column="id"\ntime_seconds_column="time"\n'
            'signal_columns=["value"]\n[windows]\nlookback_seconds=10\nstride_seconds=2\n'
        )
        self.assertEqual(config.dataset.signal_columns, ("value",))
        self.assertEqual(config.windows.stride_seconds, 2)

    def test_invalid_configuration_fails_early(self):
        cases = [
            "sead=1",
            "seed=true",
            "seed=-1",
            'log_level="silent"',
            "[windows]\nlookback_seconds=-1",
            "[windows]\nlookback_seconds=nan",
            "[windows]\nstride_seconds=true",
            "[dataset]\nsignal_columns=1",
            '[dataset]\nsubject_id_column="id"\nsignal_columns=["id"]',
            '[paths]\nraw=""',
            "dataset=1",
            "[paths]\nunknown=1",
        ]
        for text in cases:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.load(text)

    def test_manifest_serializes_config_and_runtime(self):
        config = self.load("seed=7")
        manifest = self.root / "artifacts/run.json"
        write_run_manifest(manifest, config)
        payload = json.loads(manifest.read_text())
        self.assertEqual(payload["config"]["seed"], 7)
        self.assertIn("numpy", payload["packages"])
        self.assertEqual(payload["config"]["paths"]["raw"], str(config.paths.raw))


class RuntimeTests(unittest.TestCase):
    def test_seed_reproduces_python_and_numpy_draws(self):
        rng = seed_everything(123)
        first = (random.random(), np.random.random(), rng.random())
        rng = seed_everything(123)
        self.assertEqual(first, (random.random(), np.random.random(), rng.random()))

    def test_invalid_seed_rejected(self):
        for seed in (-1, True, 2**32, 1.2):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                seed_everything(seed)

    def test_logging_is_idempotent_and_does_not_change_root(self):
        root_handlers = logging.getLogger().handlers[:]
        logger = configure_logging()
        logger = configure_logging("DEBUG")
        self.assertEqual(len(logger.handlers), 1)
        self.assertFalse(logger.propagate)
        self.assertEqual(logging.getLogger().handlers, root_handlers)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()

    def test_file_logging_creates_parent_and_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logs/run.log"
            logger = configure_logging(log_file=path)
            logger.info("Synthetic infrastructure smoke check")
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                handler.close()
            self.assertIn("Synthetic infrastructure smoke check", path.read_text())
