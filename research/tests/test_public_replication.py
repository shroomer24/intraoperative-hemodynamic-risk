"""Synthetic adapter regressions; these never select cases based on outcomes."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from intraop.data.cohort_acquisition import candidate_plan
from intraop.data.protocol import TRACKS
from intraop.data.vitaldb_reader import CaseSignals
from public_replication import cohort, runner, safety
from scripts.public_replication_worker import run_model

RESEARCH = Path(__file__).resolve().parents[1]


class PublicSelectionTests(unittest.TestCase):
    def synthetic_source(self, directory):
        root = Path(directory)
        clinical = pd.DataFrame(
            {
                "caseid": range(1, 31),
                "subjectid": range(1, 31),
                "age": [40] * 30,
                "ane_type": ["General"] * 30,
                "opstart": [0] * 30,
                "opend": [900] * 30,
                "caseend": [1000] * 30,
                "casestart": [0] * 30,
                "aline1": ["Radial"] * 30,
                "outcome_should_not_be_used": range(30),
            }
        )
        clinical.to_csv(root / "clinical_data.csv", index=False)
        for name in cohort.METADATA_FILES:
            if name != "clinical_data.csv":
                (root / name).write_text("SYNTHETIC FIXTURE NOT VITALDB\n")
        return root, clinical

    def test_original_planner_without_pilot_and_repeatability(self):
        with tempfile.TemporaryDirectory() as temp:
            root, clinical = self.synthetic_source(temp)
            a, b = cohort.derive_manifest(root), cohort.derive_manifest(root)
            self.assertEqual(a, b)
            self.assertEqual(cohort.canonical_bytes(a), cohort.canonical_bytes(b))
            self.assertEqual(
                a["public_vitaldb_case_ids_in_acquisition_order"],
                candidate_plan(clinical, [], seed=42, target=12)["candidate_order"][:12],
            )
            self.assertEqual(a["selection"]["pilot_prefix"], [])
            self.assertNotIn("candidate_subject_ids", a)

    def test_outcomes_and_metadata_row_order_do_not_choose_cases(self):
        with tempfile.TemporaryDirectory() as temp:
            root, clinical = self.synthetic_source(temp)
            a = cohort.derive_manifest(root)["public_vitaldb_case_ids_in_acquisition_order"]
            clinical["outcome_should_not_be_used"] = list(reversed(range(30)))
            clinical.iloc[::-1].to_csv(root / "clinical_data.csv", index=False)
            self.assertEqual(
                a, cohort.derive_manifest(root)["public_vitaldb_case_ids_in_acquisition_order"]
            )

    def test_committed_manifest_needs_public_metadata_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root, _ = self.synthetic_source(temp)
            with self.assertRaisesRegex(ValueError, "selection changed"):
                cohort.validate_manifest(root)

    def test_reference_contracts_and_originals_unchanged(self):
        provenance = json.loads((RESEARCH / "provenance/source_manifest.json").read_text())
        for name, entry in provenance["files"].items():
            self.assertEqual(
                hashlib.sha256((RESEARCH / name).read_bytes()).hexdigest(),
                entry["published_sha256"],
            )
        self.assertNotIn("calibration_parameters.json", safety.REFERENCES)


class PublicBoundaryTests(unittest.TestCase):
    def test_local_public_input_exemption_never_permits_tracked_raw_data(self):
        from tests import test_public_provenance as publication

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "outputs/public-replication-synthetic"
            output.mkdir(parents=True)
            (output / "scope.json").write_text(json.dumps({"scope": cohort.SCOPE}))
            (output / "synthetic.vital").write_bytes(b"NOT REAL PATIENT DATA")
            test = publication.PublicationTests("test_private_roster_and_raw_data_absent")
            with patch.object(publication, "ROOT", root):
                for tracked, ignored, allowed in [(1, 0, True), (0, 0, False), (1, 1, False)]:
                    with patch.object(
                        publication.subprocess,
                        "run",
                        side_effect=[
                            SimpleNamespace(returncode=tracked),
                            SimpleNamespace(returncode=ignored),
                        ],
                    ):
                        if allowed:
                            test.test_private_roster_and_raw_data_absent()
                        else:
                            with self.assertRaises(AssertionError):
                                test.test_private_roster_and_raw_data_absent()

    def test_virtualenv_inside_checkout_is_a_runtime_dependency_not_private_input(self):
        code = """
import sys
from pathlib import Path
import public_replication.safety as s
root=Path(sys.argv[1]).resolve()
env=root/'.venv-public'
env.mkdir()
file=env/'runtime-fixture.txt'
file.write_text('synthetic runtime dependency')
output=root/'outputs/public-replication'
output.mkdir(parents=True)
s.RESEARCH=root
sys.prefix=str(env)
s.install_io_boundary(output)
assert file.read_text()=='synthetic runtime dependency'
try: (root/'private/cohort-selection.json').read_bytes()
except PermissionError: pass
else: raise AssertionError('Private read permitted')
"""
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run(
                [sys.executable, "-B", "-c", code, temp], cwd=RESEARCH, capture_output=True
            )
        self.assertEqual(result.returncode, 0)

    def test_private_external_existing_and_symlink_output_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            (root / "outputs").mkdir()
            with patch.object(safety, "RESEARCH", root):
                for path in [
                    root / "private/cohort-selection.json",
                    root / "outputs/test",
                    root / "artifacts/modeling-v01/locked_execution",
                ]:
                    with self.assertRaises(PermissionError):
                        safety.owned_output(path)
                target = root / "outputs/public-replication"
                self.assertEqual(safety.owned_output(target), target)
                target.mkdir()
                with self.assertRaises(FileExistsError):
                    safety.owned_output(target)
                link = root / "outputs/public-replication-link"
                link.symlink_to(target, target_is_directory=True)
                with self.assertRaises(PermissionError):
                    safety.owned_output(link)

    def test_io_boundary_rejects_private_calibration_approvals_and_external_read(self):
        # Audit hooks are permanent, so exercise one in its own Python process.
        code = """
from pathlib import Path
import sys
from public_replication.safety import install_io_boundary
from public_replication.cohort import RESEARCH
output=RESEARCH/'outputs/public-replication-boundary-test'
install_io_boundary(output)
for p in [RESEARCH/'private/cohort-selection.json',
          RESEARCH/'reference/calibration_parameters.json',
          RESEARCH/'artifacts/modeling-v01/locked_execution/test/completion_receipt.json',
          RESEARCH/'artifacts/modeling-v01/locked_execution/approvals/test.json',
          Path('/undeclared-scientific-root/metadata.csv')]:
 try: p.read_bytes()
 except PermissionError: continue
 raise AssertionError('Private/held-out path was not denied before file opening')
assert (RESEARCH/'reference/feature_schema.json').read_bytes()
try: (RESEARCH/'README.md').write_text('overwrite')
except PermissionError: pass
else: raise AssertionError('Write outside output permitted')
"""
        r = subprocess.run([sys.executable, "-B", "-c", code], cwd=RESEARCH, capture_output=True)
        self.assertEqual(r.returncode, 0, "Boundary child failed; no private values displayed")

    def test_reuse_cannot_read_private_source(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(PermissionError):
                runner.copy_public_source(Path(temp) / "private", Path(temp) / "dest")

    def test_absent_authentication_is_skip_not_hosted_failure(self):
        with tempfile.TemporaryDirectory() as temp:

            def complete_conventional(command, **kwargs):
                name = command[-1]
                self.assertNotIn("tabpfn", name)
                runner.save(
                    Path(temp) / "models" / name / "completion.json",
                    {"status": "PASS", "scope": cohort.SCOPE},
                )
                return SimpleNamespace(returncode=0)

            with (
                patch.dict("os.environ", {}, clear=True),
                patch.object(runner.subprocess, "run", side_effect=complete_conventional),
            ):
                statuses = runner.execute_models(Path(temp), False, hosted=True)
            self.assertEqual(statuses["tabpfn_map"]["status"], "SKIPPED")
            self.assertEqual(statuses["tabpfn_full"]["status"], "SKIPPED")
            self.assertIn("authenticated", statuses["tabpfn_map"]["reason"])


class PublicScienceTests(unittest.TestCase):
    def test_original_causal_pipeline_splits_and_full_map_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            clinical = pd.DataFrame(
                {
                    "caseid": range(1, 13),
                    "subjectid": [1, 1, *range(2, 12)],
                    "age": [40] * 12,
                    "ane_type": ["General"] * 12,
                    "opstart": [0] * 12,
                    "opend": [1200] * 12,
                    "caseend": [1300] * 12,
                    "casestart": [0] * 12,
                }
            )
            clinical.to_csv(root / "clinical_data.csv", index=False)

            def read(source, row):
                samples = {}
                for signal in TRACKS:
                    values = np.full(1201, 80.0)
                    if signal == "map":
                        values[700:780] = 60
                        values[1100:1110] = np.nan
                    samples[signal] = pd.DataFrame(
                        {"time_seconds": np.arange(1201, dtype=float), "value": values}
                    )
                return CaseSignals(
                    str(int(row.caseid)),
                    str(int(row.subjectid)),
                    0,
                    1200,
                    samples,
                    {"tracks": {s: {"track_present": True} for s in TRACKS}},
                )

            manifest = {"public_vitaldb_case_ids_in_acquisition_order": list(range(1, 13))}
            with patch("intraop.data.pipeline.read_local_case", side_effect=read):
                a, _ = runner.build_public_tables(root, root / "a", manifest)
                b, _ = runner.build_public_tables(root, root / "b", manifest)
            self.assertEqual(a, b)
            self.assertTrue(a["subject_intersections_empty"])
            self.assertGreater(a["censored_anchors"], 0)
            self.assertGreater(a["confirmed_episodes"], 0)
            for name in (root / "a").iterdir():
                self.assertEqual(name.read_bytes(), (root / "b" / name.name).read_bytes())
            frame = pd.read_csv(root / "a/public_replication_features.csv")
            self.assertEqual(frame.columns.tolist(), list(runner.FEATURE_NAMES))
            self.assertEqual(
                pd.read_csv(root / "a/public_replication_map_features.csv").shape[1], 18
            )

    def test_mocked_tabpfn_exact_constructor_and_predictor_shapes(self):
        fake = ModuleType("tabpfn_client")
        calls = []

        class Client:
            classes_ = np.array([0, 1])

            def __init__(self, **kwargs):
                calls.append(kwargs)

            def fit(self, X, y):
                self.width = X.shape[1]
                self.assertion = len(X) == len(y)

            def predict_proba(self, X):
                assert self.width == X.shape[1] and self.assertion
                calls.append(X.shape)
                return np.tile([0.7, 0.3], (len(X), 1))

        fake.TabPFNClassifier = Client
        payload = {
            "train_X": np.zeros((8, 74)),
            "train_y": np.tile([0, 1], 4),
            "query_X": np.zeros((3, 74)),
        }
        config = json.loads((RESEARCH / "reference/selected_models.json").read_text())
        with patch.dict(sys.modules, {"tabpfn_client": fake}):
            for name in ["tabpfn_map", "tabpfn_full"]:
                scores, info = run_model(name, payload, config)
                np.testing.assert_array_equal(scores, [0.3] * 3)
                self.assertEqual(info["positive_class_column"], 1)
        self.assertEqual(
            calls,
            [
                {"model_path": "v3.5_default", "random_state": 42},
                (3, 18),
                {"model_path": "v3.5_default", "random_state": 42},
                (3, 74),
            ],
        )
