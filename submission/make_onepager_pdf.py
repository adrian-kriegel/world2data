"""Generate a one-page PDF from submission/one_pager.md using matplotlib.

Usage:
  python submission/make_onepager_pdf.py
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT_MD = ROOT / "submission" / "one_pager.md"
OUTPUT_PDF = ROOT / "submission" / "World2Data_OnePager.pdf"


def _prepare_text(src: str, width: int = 98) -> str:
    lines = []
    for raw in src.splitlines():
        if not raw.strip():
            lines.append("")
            continue
        if raw.startswith("#"):
            header = raw.lstrip("#").strip().upper()
            lines.append(header)
            continue
        wrapped = textwrap.wrap(raw, width=width, break_long_words=False, replace_whitespace=False)
        lines.extend(wrapped if wrapped else [""])
    return "\n".join(lines)


def main() -> int:
    if not INPUT_MD.exists():
        print(f"FAIL: Missing {INPUT_MD}")
        return 2

    text = INPUT_MD.read_text(encoding="utf-8")
    body = _prepare_text(text, width=98)

    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    fig.patch.set_facecolor("white")
    fig.text(
        0.06,
        0.96,
        body,
        va="top",
        ha="left",
        family="DejaVu Sans",
        fontsize=9.5,
        linespacing=1.25,
        wrap=True,
    )
    plt.axis("off")
    fig.savefig(OUTPUT_PDF, format="pdf", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"PASS: Created {OUTPUT_PDF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
