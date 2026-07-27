#!/usr/bin/env python3
"""Append an audio row to the NAVA static A/B page if it is not already present."""
import argparse
import html
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index", help="path to audio_demo/index.html")
    ap.add_argument("wav", help="WAV filename or path; only the basename is used in src")
    ap.add_argument("label", help="human-readable row label")
    ap.add_argument("--id", required=True, help="row id/number")
    args = ap.parse_args()

    index = Path(args.index)
    src = Path(args.wav).name
    text = index.read_text(encoding="utf-8") if index.exists() else "<html><body>\n"
    if f'src="{html.escape(src, quote=True)}"' in text or f"src='{html.escape(src, quote=True)}'" in text:
        print(f"{src} already listed in {index}")
        return

    row = (
        f'\n<tr><td>{html.escape(args.id)}</td><td>{html.escape(args.label)}</td>'
        f'<td><audio controls src="{html.escape(src, quote=True)}"></audio></td></tr>\n'
    )
    index.write_text(text.rstrip() + row, encoding="utf-8")
    print(f"added {src} to {index}")


if __name__ == "__main__":
    main()
