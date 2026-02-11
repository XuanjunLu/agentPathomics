# WSI Viewer Demo (Flask + OpenSeadragon)

This is a simple demo that serves Whole Slide Images (SVS/TIFF) using OpenSlide DeepZoom (DZI) and displays them in the browser using OpenSeadragon.

## Features
- Mouse wheel zoom, left-click drag to pan (OpenSeadragon)
- Thumbnail navigator
- Scale bar (based on `openslide.mpp-x` property, if present)
- DZI/tiles served by Flask via `openslide` DeepZoomGenerator
- Optional overlay: **nuclei semantic segmentation** (`nuclei_seg` class mask) with toggle + opacity

## Requirements
- Python 3.8+
- OpenSlide binaries installed on your system (Windows users: download OpenSlide for Windows and ensure the folder with `libopenslide-0.dll` is on your PATH)

Python packages:

```bash
pip install -r requirements.txt
```

## Setup
2. Place your `.svs` / `.tif` files in the `slides/` directory (create it if missing), OR set the environment variable `WSI_SLIDE_DIR` to point to an existing folder containing slides (e.g. `D:\Dataset\WSI`).
3. Run:

```bash
# Option 1: use default slides/ directory
python app.py

# Option 2: point to your slides folder (Windows PowerShell)
$env:WSI_SLIDE_DIR='D:\Dataset\WSI'; python app.py
```
3. Open `http://localhost:5000` in your browser. The web UI includes a **Slides** management panel where you can upload new `.svs/.tif` files, view thumbnails, open or delete slides.

## Notes
- If a slide does not expose `openslide.mpp-x`, the scale bar will be hidden.
- For production use, serve tiles via a production WSGI server (gunicorn/uwsgi) and front with NGINX, add caching for tiles.
- This demo includes a simple management UI for uploading and deleting slides; uploads are not authenticated and should only be used in a trusted environment or behind access controls in production.
- Annotation tools: the right-side panel provides **Point**, **Rectangle**, and **Polygon** annotations. Annotations are stored in image (absolute) coordinates and will scale/position correctly when zooming/panning. Use **Save JSON** to download annotations and **Load JSON** to import them. JSON contains {slide, annotations} where annotations list type and coordinates.
- Overlay results directory:
  - By default the app looks for a sibling folder `../result/` relative to this project folder.
  - You can override with environment variable `WSI_RESULT_DIR`.
  - Nuclei overlay expects files: `result/nuclei_seg/<slide_stem>_class.png` (same stem as the uploaded WSI filename without extension). The app will build a cached preview into `result/nuclei_seg_preview/` on first use.
  - For **fast, SVS-like loading**, you can pre-generate a DeepZoom pyramid for the mask:
    - Output: `result/nuclei_seg_dzi/<slide_stem>_class.dzi` + `<slide_stem>_class_files/`
    - The viewer will automatically prefer the DZI overlay when present.

## Generate mask DZI (recommended)
Install libvips and ensure `vips` is on your PATH, then run:

```bash
python tools/make_mask_dzi.py --input "..\\result\\nuclei_seg\\<slide_stem>_class.png" --output-dir "..\\result\\nuclei_seg_dzi"
```

This produces `..\\result\\nuclei_seg_dzi\\<slide_stem>_class.dzi` and a `..._files\\` folder with PNG tiles.

## License
MIT
