#!/usr/bin/env python3
"""Compose three-page event PDFs into one three-column PDF per event."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from PIL import Image


def render_three_pages(source: str, prefix: str, dpi: int) -> list[Image.Image]:
    subprocess.run([
        "pdftoppm", "-f", "1", "-l", "3", "-png", "-r", str(dpi),
        source, prefix,
    ], check=True)
    paths = sorted(Path(prefix).parent.glob(Path(prefix).name + "-*.png"))
    if len(paths) != 3:
        raise RuntimeError(f"Expected three pages in {source}, found {len(paths)}")
    return [Image.open(path).convert("RGB") for path in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-pdf", action="append", required=True)
    parser.add_argument("--output-pdf", required=True)
    parser.add_argument("--output-audit", required=True)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()
    output = Path(args.output_pdf)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined, records = [], []
    with tempfile.TemporaryDirectory(prefix="detid_shift_three_way_") as workdir:
        for index, source in enumerate(args.event_pdf, start=1):
            pages = render_three_pages(
                source, os.path.join(workdir, f"event_{index:04d}"), args.dpi,
            )
            sizes = {page.size for page in pages}
            if len(sizes) != 1:
                raise RuntimeError(f"Three page dimensions differ for {source}: {sizes}")
            width, height = pages[0].size
            canvas = Image.new("RGB", (3 * width, height), "white")
            for column, page in enumerate(pages):
                canvas.paste(page, (column * width, 0))
            combined.append(canvas)
            records.append({
                "event_pdf": str(Path(source).resolve()),
                "column_pixels": [width, height],
                "combined_pixels": [3 * width, height],
                "column_aspect_ratios": [page.width / page.height for page in pages],
                "scaling_applied": False,
            })
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".pdf", dir=output.parent,
        )
        os.close(descriptor)
        try:
            combined[0].save(
                temporary, "PDF", resolution=float(args.dpi), save_all=True,
                append_images=combined[1:],
            )
            os.replace(temporary, output)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    audit_path = Path(args.output_audit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps({
        "layout": "original_track_transform_simple_transform",
        "aspect_ratio_preserved": True,
        "dpi": args.dpi,
        "page_count": len(records),
        "pages": records,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} three-column event pages to {output}")


if __name__ == "__main__":
    main()
