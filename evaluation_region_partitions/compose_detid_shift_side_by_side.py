#!/usr/bin/env python3
"""Combine two-page event PDFs into one original-left/transformed-right PDF."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from PIL import Image


def render_pdf_pages(source: str, prefix: str, dpi: int) -> tuple[Image.Image, Image.Image]:
    subprocess.run([
        "pdftoppm", "-f", "1", "-l", "2", "-png", "-r", str(dpi),
        source, prefix,
    ], check=True)
    pages = sorted(Path(prefix).parent.glob(Path(prefix).name + "-*.png"))
    if len(pages) != 2:
        raise RuntimeError(f"Expected two pages in {source}, found {len(pages)}")
    return tuple(Image.open(page).convert("RGB") for page in pages)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-pdf", action="append", required=True)
    parser.add_argument("--output-pdf", required=True)
    parser.add_argument("--output-audit", required=True)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()
    output = Path(args.output_pdf)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined_pages, records = [], []
    with tempfile.TemporaryDirectory(prefix="detid_shift_side_by_side_") as workdir:
        for index, source in enumerate(args.event_pdf, start=1):
            original, transformed = render_pdf_pages(
                source, os.path.join(workdir, f"event_{index:04d}"), args.dpi,
            )
            if original.size != transformed.size:
                raise RuntimeError(
                    f"Page dimensions differ for {source}: "
                    f"{original.size} versus {transformed.size}"
                )
            canvas = Image.new(
                "RGB", (original.width + transformed.width, original.height), "white"
            )
            canvas.paste(original, (0, 0))
            canvas.paste(transformed, (original.width, 0))
            combined_pages.append(canvas)
            records.append({
                "event_pdf": str(Path(source).resolve()),
                "original_pixels": list(original.size),
                "transformed_pixels": list(transformed.size),
                "combined_pixels": list(canvas.size),
                "original_aspect_ratio": original.width / original.height,
                "transformed_aspect_ratio": transformed.width / transformed.height,
                "scaling_applied": False,
            })
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".pdf", dir=output.parent,
        )
        os.close(descriptor)
        try:
            combined_pages[0].save(
                temporary, "PDF", resolution=float(args.dpi), save_all=True,
                append_images=combined_pages[1:],
            )
            os.replace(temporary, output)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    audit_output = Path(args.output_audit)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_output, "w", encoding="utf-8") as stream:
        json.dump({
            "layout": "original_left_transformed_right",
            "aspect_ratio_preserved": True,
            "dpi": args.dpi,
            "page_count": len(records),
            "pages": records,
        }, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Wrote {len(records)} side-by-side event pages to {output}")


if __name__ == "__main__":
    main()
