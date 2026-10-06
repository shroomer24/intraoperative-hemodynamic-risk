"""Fail-closed audit of the static upload only. Never output matched secret bytes."""

from pathlib import Path
import argparse
import hashlib
import json
import re

PATTERNS = {
    "local_absolute_path": rb"/Users/|/Library/Frameworks/|/private/(?:tmp|var)/|/home/[A-Za-z0-9_-]+/",
    "credential_file_or_environment": rb"~/.tabpfn\.env|TABPFN_TOKEN|CLOUDFLARE_API_TOKEN|CLOUDFLARE_API_KEY",
    "private_audit_or_scientific_path": rb"source_crosswalk|replay-private|locked_execution/test|intraop-prediction/|\.codex/|phase[0-9]+-review/|continue-prompt-3b-from-the-resolved",
    "loopback": rb"localhost|127\.0\.0\.1",
    "credential_literal": rb"(?i)bearer\s+[A-Za-z0-9._~-]{16,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|sk-[A-Za-z0-9]{24,}|AKIA[0-9A-Z]{16}|eyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}",
}
RELEASE_FILES = {
    "index.html",
    "demo.html",
    "_headers",
    "_redirects",
    "404.html",
    "not-found.css",
    "favicon.svg",
    "favicon.ico",
    "apple-touch-icon.png",
    "social-preview-v01.png",
    "THIRD_PARTY_NOTICES.txt",
}
REPLAY_FILES = {"replay-v01/manifest.json", "replay-v01/cases/index.json"} | {
    f"replay-v01/cases/case-{n:03d}/{name}.json"
    for n in range(1, 23)
    for name in ("signals", "prediction-windows", "events")
}


def findings_for_bytes(data):
    return [
        category for category, pattern in PATTERNS.items() if re.search(pattern, data)
    ]


def audit(dist, public):
    hashes = {}
    findings = []
    if not dist.is_dir():
        return {"status": "FAIL", "findings": [{"category": "missing_build"}]}
    for path in sorted(dist.rglob("*")):
        if path.is_symlink():
            findings.append(
                {"file": path.relative_to(dist).as_posix(), "category": "symlink"}
            )
            continue
        if not path.is_file():
            continue
        name = path.relative_to(dist).as_posix()
        data = path.read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        if name not in RELEASE_FILES | REPLAY_FILES and not re.fullmatch(
            r"assets/[A-Za-z0-9_-]+-[A-Za-z0-9_-]{8,}\.(js|css)", name
        ):
            findings.append({"file": name, "category": "not_upload_allowlisted"})
        findings.extend(
            {"file": name, "category": category}
            for category in findings_for_bytes(data)
        )
        if name in REPLAY_FILES and data != (public / name).read_bytes():
            findings.append({"file": name, "category": "frozen_replay_diff"})
        if len(data) > 25 * 1024 * 1024:
            findings.append({"file": name, "category": "pages_file_size_limit"})
    for name in sorted((RELEASE_FILES | REPLAY_FILES) - hashes.keys()):
        findings.append({"file": name, "category": "missing_required_asset"})
    if not any(name.endswith(".js") for name in hashes) or not any(
        name.endswith(".css") and name.startswith("assets/") for name in hashes
    ):
        findings.append({"category": "missing_app_bundle"})
    digest = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "status": "PASS" if not findings else "FAIL",
        "schema_version": 1,
        "build_sha256": digest,
        "file_count": len(hashes),
        "files_sha256": hashes,
        "approved_replay_assets": len(REPLAY_FILES),
        "required_release_assets": len(RELEASE_FILES),
        "source_maps": 0 if not any(x.endswith(".map") for x in hashes) else 1,
        "scan_categories": list(PATTERNS),
        "findings": findings,
        "matched_values_retained": False,
        "credential_files_read": False,
        "source_scientific_tables_read": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    app = Path(__file__).resolve().parent.parent
    result = audit(app / "dist", app / "public")
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "build_sha256", "file_count", "findings")
                if key in result
            }
        )
    )
    raise SystemExit(0 if result["status"] == "PASS" else 1)
