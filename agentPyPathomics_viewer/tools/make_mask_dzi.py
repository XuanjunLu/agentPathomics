"""
Generate DeepZoom (DZI + _files tiles) for segmentation masks.

Why:
  - SVS/WSI loads fast because it's a tiled pyramid (OpenSeadragon requests only needed tiles).
  - A single huge PNG mask is slow to preview because PNG decoding is scanline-based.

This script uses libvips (recommended) to build a DeepZoom pyramid efficiently, preserving
mask label values by using nearest-neighbor shrink.

Prerequisite (Windows):
  - You need libvips runtime (DLLs). Two equivalent options:
    1) Install libvips and make sure `vips` is on PATH
    2) OR download libvips zip, then pass `--vips-bin <...\\vips\\bin>` (or set env VIPS_BIN)

  Windows binaries: https://www.libvips.org/install.html (download binaries)

Example:
  python tools/make_mask_dzi.py ^
    --input "..\\result\\nuclei_seg\\TCGA-XXX_class.png" ^
    --output-dir "..\\result\\nuclei_seg_dzi" ^
    --name "TCGA-XXX_class" ^
    --tile-size 256

It will produce:
  <output-dir>/<name>.dzi
  <output-dir>/<name>_files/<level>/<col>_<row>.png

NOTE:
  If you generate DZI directly from a raw label mask (values 0..5), the overlay can look
  "almost black" in the browser because values 1..5 are very dark in grayscale.
  Use --colorize to map classes to colors + transparent background before dzsave.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import List


def _which_vips() -> str | None:
    # Prefer explicit env var if provided
    env = os.environ.get("VIPS_EXE")
    if env and os.path.exists(env):
        return env
    return shutil.which("vips")

def _add_vips_bin_to_dll_search(vips_bin: str | None) -> None:
    # Helps pyvips find libvips DLLs without requiring global PATH changes.
    if not vips_bin:
        return
    try:
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(vips_bin)
    except Exception:
        pass


def _run(cmd: List[str]) -> None:
    print("[mask-dzi] running:", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.stdout:
        print(p.stdout, flush=True)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=False, help="Input mask image (e.g. *_class.png)")
    ap.add_argument("--output-dir", required=True, help="Directory to write <name>.dzi and <name>_files/")
    ap.add_argument("--name", required=False, help="Output base name (without extension). Default: input stem")
    ap.add_argument("--tile-size", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=0)
    ap.add_argument(
        "--depth",
        default="onepixel",
        help="Pyramid depth for dzsave. Use onepixel (full DeepZoom pyramid, recommended for OpenSeadragon), onetile (stop early), or one (single level). Default: onepixel",
    )
    ap.add_argument(
        "--colorize",
        action="store_true",
        help="Colorize semantic labels (0..5) to RGBA (0=transparent) before generating DZI. Requires pyvips.",
    )
    ap.add_argument(
        "--progress",
        action="store_true",
        help="Show libvips progress (CLI backend only).",
    )
    ap.add_argument(
        "--auto",
        action="store_true",
        help="Auto-pick slide+mask from a directory (default: ./svs). Ignores --name and uses <svs_stem>_class.",
    )
    ap.add_argument("--svs-dir", default="svs", help="Directory containing example .svs and mask (used with --auto).")
    ap.add_argument(
        "--vips-bin",
        default=os.environ.get("VIPS_BIN"),
        help="Directory containing vips DLLs (typically ...\\vips\\bin). Optional; helps pyvips on Windows.",
    )
    args = ap.parse_args()

    if args.auto:
        import pathlib

        svs_dir = pathlib.Path(args.svs_dir)
        if not svs_dir.exists():
            print(f"[mask-dzi] --svs-dir not found: {svs_dir.resolve()}", file=sys.stderr)
            return 2
        svs_list = sorted(svs_dir.glob("*.svs"))
        if not svs_list:
            print(f"[mask-dzi] No .svs found in: {svs_dir.resolve()}", file=sys.stderr)
            return 2
        svs = svs_list[0]
        stem = svs.stem
        # Prefer conventional nuclei semantic mask name
        mask = svs_dir / f"{stem}_class.png"
        if not mask.exists():
            pngs = sorted(svs_dir.glob("*.png"))
            if not pngs:
                print(f"[mask-dzi] No .png mask found in: {svs_dir.resolve()}", file=sys.stderr)
                return 2
            mask = pngs[0]
        in_path = str(mask.resolve())
        name = f"{stem}_class"
        print(f"[mask-dzi] auto: svs={svs.name} mask={os.path.basename(in_path)} name={name}", flush=True)
    else:
        if not args.input:
            print("[mask-dzi] --input is required unless you use --auto", file=sys.stderr)
            return 2
        in_path = os.path.abspath(args.input)
        if not os.path.exists(in_path):
            print(f"[mask-dzi] input not found: {in_path}", file=sys.stderr)
            return 2
        name = args.name or os.path.splitext(os.path.basename(in_path))[0]

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    out_base = os.path.join(out_dir, name)  # libvips writes out_base.dzi and out_base_files/

    vips = _which_vips()
    if vips and not args.colorize:
        print(f"[mask-dzi] vips: {vips}", flush=True)
        # IMPORTANT for masks:
        # - use PNG tiles (lossless) so labels stay exact
        # - use nearest shrink when generating pyramid to avoid creating non-integer label blends
        cmd = [vips]
        if args.progress:
            cmd += ["--vips-progress"]
        cmd += [
            "dzsave",
            in_path,
            out_base,
            "--layout",
            "dz",
            "--suffix",
            ".png",
            "--tile-size",
            str(int(args.tile_size)),
            "--overlap",
            str(int(args.overlap)),
            "--depth",
            str(args.depth),
            "--region-shrink",
            "nearest",
        ]
        _run(cmd)
    else:
        # Fallback: use pyvips (still needs libvips DLLs available).
        _add_vips_bin_to_dll_search(args.vips_bin)
        try:
            import pyvips  # type: ignore
        except Exception as e:
            print(
                "[mask-dzi] Cannot find `vips` executable, and cannot import `pyvips`.\n"
                "Fix options:\n"
                "  - Install libvips and ensure `vips` is on PATH, OR set VIPS_EXE\n"
                "  - OR `pip install pyvips` AND ensure libvips DLLs are discoverable:\n"
                "      - add <vips>\\bin to PATH, OR pass --vips-bin <vips>\\bin (or set VIPS_BIN)\n"
                "Windows binaries: https://www.libvips.org/install.html\n"
                f"Import error: {e}",
                file=sys.stderr,
            )
            return 3

        print("[mask-dzi] using pyvips backend", flush=True)
        try:
            img = pyvips.Image.new_from_file(in_path, access="sequential")
            if args.colorize:
                # Semantic palette (same as the viewer legend)
                # 0 background -> transparent
                palette = {
                    1: (230, 25, 75),     # Neoplastic
                    2: (0, 130, 200),     # Inflammatory
                    3: (60, 180, 75),     # Connective
                    4: (128, 128, 128),   # Dead
                    5: (255, 225, 25),    # Epithelial
                }
                # Ensure uchar
                img = img.cast("uchar")
                # Build alpha: 255 where label>0 else 0
                alpha = (img > 0).ifthenelse(255, 0).cast("uchar")

                def _class_mask(v: int):
                    # 1 where img==v else 0 (uchar)
                    return (img == v).ifthenelse(1, 0).cast("uchar")

                r = pyvips.Image.black(img.width, img.height).cast("uchar")
                g = r.copy()
                b = r.copy()
                for cls, (cr, cg, cb) in palette.items():
                    m = _class_mask(cls)
                    # m is 0/1, scale to channel
                    r = (r + (m * cr)).cast("uchar")
                    g = (g + (m * cg)).cast("uchar")
                    b = (b + (m * cb)).cast("uchar")

                rgba = r.bandjoin([g, b, alpha]).copy(interpretation="srgb")
                rgba.dzsave(
                    out_base,
                    layout="dz",
                    suffix=".png",
                    tile_size=int(args.tile_size),
                    overlap=int(args.overlap),
                    depth=str(args.depth),
                    region_shrink="nearest",
                )
            else:
                # Grayscale output (labels) - will look dark if labels are small numbers
                img.dzsave(
                    out_base,
                    layout="dz",
                    suffix=".png",
                    tile_size=int(args.tile_size),
                    overlap=int(args.overlap),
                    depth=str(args.depth),
                    region_shrink="nearest",
                )
        except Exception as e:
            print(
                "[mask-dzi] pyvips dzsave failed.\n"
                "If you see DLL errors (e.g. libvips-*.dll missing), install libvips Windows binaries and either:\n"
                "  - add <vips>\\bin to PATH, or\n"
                "  - rerun with --vips-bin <vips>\\bin\n"
                f"Error: {e}",
                file=sys.stderr,
            )
            return 4

    print("[mask-dzi] done:")
    print("  ", out_base + ".dzi")
    print("  ", out_base + "_files/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

