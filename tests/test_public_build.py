import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "audit", Path(__file__).resolve().parents[1] / "tools/audit_public_build.py"
)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class PublicBuildTests(unittest.TestCase):
    def tree(self, root):
        dist = root / "dist"
        public = root / "public"
        for name in (
            a.RELEASE_FILES
            | a.REPLAY_FILES
            | {"assets/index-12345678.js", "assets/index-12345678.css"}
        ):
            p = dist / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"approved static data")
            if name in a.REPLAY_FILES:
                p = public / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"approved static data")
        return dist, public

    def test_safe_exact_allowlist_passes(self):
        with tempfile.TemporaryDirectory() as t:
            d, p = self.tree(Path(t))
            self.assertEqual(a.audit(d, p)["status"], "PASS")

    def test_categories_never_retain_matched_values(self):
        for text in [
            b"/Users/private/person",
            b"localhost",
            b"TABPFN_TOKEN",
            b"Bearer " + b"X" * 32,
            b"source_crosswalk.json",
        ]:
            self.assertTrue(a.findings_for_bytes(text))
            self.assertNotIn(text.decode(), json.dumps(a.findings_for_bytes(text)))

    def test_unapproved_files_and_source_maps_block(self):
        with tempfile.TemporaryDirectory() as t:
            d, p = self.tree(Path(t))
            (d / ".env").write_text("blocked")
            (d / "assets/index.js.map").write_text("{}")
            self.assertEqual(a.audit(d, p)["status"], "FAIL")

    def test_frozen_replay_diff_blocks(self):
        with tempfile.TemporaryDirectory() as t:
            d, p = self.tree(Path(t))
            (d / "replay-v01/manifest.json").write_text("changed")
            self.assertIn(
                "frozen_replay_diff", [x["category"] for x in a.audit(d, p)["findings"]]
            )

    def test_missing_asset_and_symlink_block(self):
        with tempfile.TemporaryDirectory() as t:
            d, p = self.tree(Path(t))
            (d / "favicon.svg").unlink()
            (d / "forbidden").symlink_to(p)
            self.assertEqual(a.audit(d, p)["status"], "FAIL")

    def test_file_digest_changes_release_identity(self):
        with tempfile.TemporaryDirectory() as t:
            d, p = self.tree(Path(t))
            first = a.audit(d, p)["build_sha256"]
            (d / "index.html").write_text("new safe release")
            self.assertNotEqual(first, a.audit(d, p)["build_sha256"])


if __name__ == "__main__":
    unittest.main()
