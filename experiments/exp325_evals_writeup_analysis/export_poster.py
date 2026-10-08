"""Package every static figure as vector PDF/SVG and a poster-size PNG.

Rasterize the freshly rendered vector PDFs, never upscale the web PNGs.
Default: 7,200 pixels wide, tagged 300 dpi (24 inches / 61 cm wide).
Large exports stay in scratch and the public bucket; the inventory stays in git.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
DESTINATION = "hf://buckets/open-athena/MarinFold/data/exp325-writeup-analysis/poster-2026-10-08"


def digest(path: Path) -> str:
    """Hash an artifact without loading the entire file into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    """Create a print bundle, including numerical inputs and per-plot lineage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width-px", type=int, default=7200)
    parser.add_argument("--print-dpi", type=int, default=300)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    if args.width_px < 1 or args.print_dpi < 1:
        raise ValueError("Pixel width and print resolution must be positive")
    stage = HERE / "scratch/poster_public"
    bundle = stage / "poster_plots"
    if stage.exists():
        shutil.rmtree(stage)
    bundle.mkdir(parents=True)
    records = []
    for source in sorted((HERE / "plots").glob("*.png")):
        name = source.stem
        for extension in ("pdf", "svg"):
            vector = source.with_suffix(f".{extension}")
            if not vector.exists():
                raise FileNotFoundError(f"Rerun render.py; missing native vector: {vector}")
            output = bundle / extension / vector.name
            output.parent.mkdir(exist_ok=True)
            shutil.copyfile(vector, output)
        png = bundle / "png" / source.name
        png.parent.mkdir(exist_ok=True)
        subprocess.run(["pdftocairo", "-png", "-singlefile", "-r", str(args.print_dpi),
                        "-scale-to-x", str(args.width_px), "-scale-to-y", "-1",
                        str(source.with_suffix(".pdf")), str(png.with_suffix(""))], check=True)
        with Image.open(png) as raster:
            width, height = raster.size
            raster.save(png, dpi=(args.print_dpi, args.print_dpi))
        metadata = json.loads(source.with_suffix(".png.meta.json").read_text())
        record = dict(name=name, width_px=width, height_px=height, dpi=args.print_dpi,
                      print_width_inches=width / args.print_dpi, print_height_inches=height / args.print_dpi,
                      source_pdf_sha256=digest(source.with_suffix(".pdf")),
                      png_sha256=digest(png), **metadata)
        records.append(record)
        print(f"{name}: {width} × {height}", flush=True)
    for font in (HERE / "data/inputs").glob("Lato-*"):
        output = bundle / "fonts" / font.name
        output.parent.mkdir(exist_ok=True)
        shutil.copyfile(font, output)
    for filename in ("POSTER.md", "FIGURES.md", "PL5_ANALYSIS.md"):
        shutil.copyfile(HERE / filename, bundle / filename)
    manifest = dict(destination=DESTINATION, generator="export_poster.py", width_px=args.width_px,
                    print_dpi=args.print_dpi, figure_count=len(records), figures=records)
    (bundle / "poster_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (HERE / "data/poster_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(stage / "poster_plots.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(stage))
    # Publish the small analysis/code as well as print artwork. Source snapshots
    # make the baseline joins reproducible; large rollout pools already live at
    # the durable public URLs in pl5_analysis.json.
    for pattern in ("data/pl5*", "data/inputs/pl5*", "*pl5*.py", "export_poster.py", "render*.py",
                    "prepare.py", "theme.py", "build_summary.py", "pyproject.toml", "uv.lock",
                    "generation/score_contacts.py", "plots/summary.pdf", "site/*.json",
                    "site/index.html", "site/plotly.min.js", "DRAFT.md", "summary_narrative.md"):
        for path in HERE.glob(pattern):
            output = stage / path.relative_to(HERE)
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, output)
    pl5 = json.loads((HERE / "data/pl5_analysis.json").read_text())
    for relative in pl5["sources"]:
        source = HERE.parents[1] / relative
        output = stage / "source_tables" / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output)
    inventory = {str(p.relative_to(stage)): dict(bytes=p.stat().st_size, sha256=digest(p))
                 for p in sorted(stage.rglob("*")) if p.is_file()}
    (stage / "publication_manifest.json").write_text(json.dumps(dict(destination=DESTINATION, files=inventory), indent=2) + "\n")
    print(f"{len(records)} figures; ZIP {(stage / 'poster_plots.zip').stat().st_size / 1e6:.1f} MB")
    if args.upload:
        subprocess.run(["hf", "buckets", "sync", str(stage), DESTINATION, "--quiet"], check=True)


if __name__ == "__main__":
    main()
