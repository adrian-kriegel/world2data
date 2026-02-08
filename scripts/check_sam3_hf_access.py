"""Check whether Hugging Face access to facebook/sam3 is available.

Usage:
  python scripts/check_sam3_hf_access.py
  python scripts/check_sam3_hf_access.py --token hf_xxx
  python scripts/check_sam3_hf_access.py --repo facebook/sam3
"""

from __future__ import annotations

import argparse
import os
import sys

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError


def _get_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token
    for env_name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        val = os.environ.get(env_name)
        if val:
            return val
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check SAM3 gated-model access on Hugging Face")
    parser.add_argument("--repo", default="facebook/sam3", help="HF repo id (default: facebook/sam3)")
    parser.add_argument("--token", default=None, help="HF token (or use HF_TOKEN / HUGGINGFACE_HUB_TOKEN)")
    parser.add_argument("--filename", default="config.json", help="File to probe (default: config.json)")
    args = parser.parse_args()

    token = _get_token(args.token)
    if not token:
        print("FAIL: No Hugging Face token found.")
        print("Set one of: HF_TOKEN or HUGGINGFACE_HUB_TOKEN")
        print("Then run again.")
        print()
        print("How to get access:")
        print("1. Sign in on https://huggingface.co")
        print("2. Open https://huggingface.co/facebook/sam3")
        print("3. Click 'Request access' / accept model terms")
        print("4. Create a token with Read scope: https://huggingface.co/settings/tokens")
        print("5. In terminal: set HF_TOKEN=<your_token>")
        return 2

    try:
        path = hf_hub_download(
            repo_id=args.repo,
            filename=args.filename,
            token=token,
            local_files_only=False,
        )
        print("PASS: SAM3 access is active.")
        print(f"Downloaded probe file: {path}")
        return 0
    except GatedRepoError:
        print(f"FAIL: Access to '{args.repo}' is still gated for this account/token.")
        print("Open the model page and request/accept access:")
        print(f"https://huggingface.co/{args.repo}")
        return 3
    except HfHubHTTPError as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        print(f"FAIL: Hugging Face HTTP error (status={status}): {e}")
        if status in (401, 403):
            print("Token is invalid, lacks permissions, or model access was not granted.")
        return 4
    except Exception as e:  # pragma: no cover - defensive fallback
        print(f"FAIL: Unexpected error: {e}")
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
