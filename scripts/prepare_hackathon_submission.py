"""Prepare a hackathon submission package and validate required artifacts.

Usage:
  python scripts/prepare_hackathon_submission.py
"""

from __future__ import annotations

import datetime as dt
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = ROOT / "submission"
DIST_DIR = ROOT / "dist"


REQUIRED_FILES = [
    SUBMISSION_DIR / "project_summary_150_300_words.txt",
    SUBMISSION_DIR / "World2Data_OnePager.pdf",
    SUBMISSION_DIR / "demo_video_script_60s.md",
    SUBMISSION_DIR / "tech_video_script_60s.md",
    SUBMISSION_DIR / "live_pitch_2min_outline.md",
]


EXCLUDE_DIR_NAMES = {
    ".git",
    ".venv",
    ".venv-world2data",
    ".pytest_cache",
    "__pycache__",
    "mast3r",
    "overnight_output",
    "demo_output",
    "dist",
}

EXCLUDE_EXTS = {
    ".pt",
    ".pth",
    ".rrd",
    ".usda",
    ".usd",
    ".usdc",
    ".ply",
    ".mp4",
    ".zip",
}


def _is_excluded(path: Path) -> bool:
    parts = set(path.parts)
    if parts.intersection(EXCLUDE_DIR_NAMES):
        return True
    if path.suffix.lower() in EXCLUDE_EXTS:
        return True
    return False


def main() -> int:
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    missing = [str(p.relative_to(ROOT)) for p in REQUIRED_FILES if not p.exists()]
    if missing:
        print("FAIL: Missing required submission artifacts:")
        for m in missing:
            print(f"  - {m}")
        print("Create those files first, then re-run this script.")
        return 2

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = DIST_DIR / f"world2data_submission_code_{timestamp}.zip"
    manifest_path = DIST_DIR / f"world2data_submission_manifest_{timestamp}.json"

    included = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in ROOT.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT)
            if _is_excluded(rel):
                continue
            zf.write(path, arcname=str(rel))
            included.append(str(rel))

    manifest = {
        "created_at": dt.datetime.now().isoformat(),
        "zip_file": str(zip_path.relative_to(ROOT)),
        "included_files_count": len(included),
        "required_artifacts": [str(p.relative_to(ROOT)) for p in REQUIRED_FILES],
        "included_files_sample": included[:200],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("PASS: Submission package prepared.")
    print(f"Code zip: {zip_path}")
    print(f"Manifest: {manifest_path}")
    print()
    print("Next: upload zip + repository link + artifacts to the two submission portals.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
