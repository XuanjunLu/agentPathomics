import io
import os
import re
import glob
from functools import lru_cache
from flask import Flask, Response, send_file, abort, jsonify, send_from_directory, request, session, redirect
try:
    from flask_caching import Cache  # optional (falls back to no-cache if missing)
except Exception:
    Cache = None
from werkzeug.utils import secure_filename
import openslide
from openslide.deepzoom import DeepZoomGenerator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Slides directory can be overridden by environment variable WSI_SLIDE_DIR
SLIDE_DIR = os.environ.get('WSI_SLIDE_DIR', os.path.join(BASE_DIR, "slides"))
# Result directory (where nuclei_seg etc live). Defaults to sibling "result/" next to this project folder.
RESULT_DIR = os.environ.get('WSI_RESULT_DIR', os.path.join(os.path.dirname(BASE_DIR), "result"))
NUCLEI_SEG_DIR = os.path.join(RESULT_DIR, "nuclei_seg")
NUCLEI_SEG_PREVIEW_DIR = os.environ.get('WSI_NUCLEI_PREVIEW_DIR', os.path.join(RESULT_DIR, "nuclei_seg_preview"))
NUCLEI_SEG_DZI_DIR = os.environ.get('WSI_NUCLEI_DZI_DIR', os.path.join(RESULT_DIR, "nuclei_seg_dzi"))
TISSUE_MASK_DIR = os.path.join(RESULT_DIR, "tissue_mask")
TISSUE_MASK_PREVIEW_DIR = os.environ.get('WSI_TISSUE_MASK_PREVIEW_DIR', os.path.join(RESULT_DIR, "tissue_mask_preview"))
TISSUE_SEG_FULL_DIR = os.path.join(RESULT_DIR, "tissue_seg", "full_images")
ALLOWED_EXT = ('.svs', '.tif', '.tiff', '.ndpi')

def _parse_rgba_env(var_name: str, default: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """
    Parse "R,G,B,A" from env into 4 ints (0-255). Falls back to default on error.
    """
    raw = os.environ.get(var_name)
    if not raw:
        return default
    try:
        parts = [int(x.strip()) for x in raw.split(',')]
        if len(parts) != 4:
            return default
        r, g, b, a = parts
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        a = max(0, min(255, a))
        return (r, g, b, a)
    except Exception:
        return default

# tissue_mask preview color (backend overlay). Default is GREEN (0,255,0) with alpha.
TISSUE_MASK_RGBA = _parse_rgba_env('WSI_TISSUE_MASK_RGBA', (0, 255, 0, 220))
# Tag is used to version disk previews + server memoize keys (so color changes take effect immediately).
TISSUE_MASK_COLOR_TAG = f"c{TISSUE_MASK_RGBA[0]}_{TISSUE_MASK_RGBA[1]}_{TISSUE_MASK_RGBA[2]}_{TISSUE_MASK_RGBA[3]}"

# Limit uploads (e.g., 10 GB) to avoid runaway uploads in demo
# Serve static assets from the conventional /static/ path.
app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024  # 10 GiB
# Cache (used for overlay previews, thumbnails, etc.). If flask-caching isn't installed, run without server cache.
if Cache:
    app.config.setdefault('CACHE_TYPE', os.environ.get('WSI_CACHE_TYPE', 'filesystem'))
    app.config.setdefault('CACHE_DIR', os.environ.get('WSI_CACHE_DIR', os.path.join(BASE_DIR, '.cache')))
    app.config.setdefault('CACHE_DEFAULT_TIMEOUT', int(os.environ.get('WSI_CACHE_TTL', '86400')))  # 1 day
    cache = Cache(app)
else:
    class _DummyCache:
        def memoize(self, *args, **kwargs):
            def deco(fn):
                return fn
            return deco
    cache = _DummyCache()
# Simple session secret for demo; in production set a strong secret via env
app.secret_key = os.environ.get('WSI_SECRET_KEY', 'dev-secret-1234')

# Configure logging to ensure detailed server-side logs
import logging
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
handler.setFormatter(formatter)
# Attach handler only if logger has no handlers (prevents duplicate logs in reloader)
if not app.logger.handlers:
    app.logger.addHandler(handler)
app.logger.setLevel(logging.DEBUG)
app.logger.debug('Logging configured for WSI viewer')


# Helper: safe filename check
_FILENAME_RE = re.compile(r"^[\w\-. ]+$")

def _safe_name(name):
    # Allow safe names and ensure they exist in SLIDE_DIR
    return bool(_FILENAME_RE.match(name)) and os.path.exists(os.path.join(SLIDE_DIR, name))

def _allowed_file(filename):
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXT

def _slide_to_result_stem(slide_name: str) -> str:
    # slide_name is the uploaded filename like "TCGA-xxx.svs"
    return os.path.splitext(os.path.basename(slide_name))[0]

def _nuclei_class_mask_path(slide_name: str) -> str:
    stem = _slide_to_result_stem(slide_name)
    return os.path.join(NUCLEI_SEG_DIR, f"{stem}_class.png")

def _has_nuclei_seg(slide_name: str) -> bool:
    try:
        return os.path.exists(_nuclei_class_mask_path(slide_name))
    except Exception:
        return False

def _nuclei_class_dzi_path(slide_name: str) -> str:
    stem = _slide_to_result_stem(slide_name)
    return os.path.join(NUCLEI_SEG_DZI_DIR, f"{stem}_class.dzi")

def _nuclei_class_dzi_tiles_dir(slide_name: str) -> str:
    stem = _slide_to_result_stem(slide_name)
    return os.path.join(NUCLEI_SEG_DZI_DIR, f"{stem}_class_files")

def _has_nuclei_seg_dzi(slide_name: str) -> bool:
    try:
        # Prefer .dzi, but allow running with only the *_files/ directory copied.
        return os.path.exists(_nuclei_class_dzi_path(slide_name)) or os.path.exists(_nuclei_class_dzi_tiles_dir(slide_name))
    except Exception:
        return False

def _tissue_mask_path(slide_name: str) -> str:
    stem = _slide_to_result_stem(slide_name)
    return os.path.join(TISSUE_MASK_DIR, f"{stem}.png")

def _tissue_mask_any_preview_path(slide_name: str, downsample: int) -> str | None:
    """
    Find any existing cached tissue_mask preview on disk, regardless of color tag.
    This enables running the viewer with only tissue_mask_preview copied (no raw tissue_mask PNG).
    """
    stem = _slide_to_result_stem(slide_name)
    ds = max(1, int(downsample))
    pat = os.path.join(TISSUE_MASK_PREVIEW_DIR, f"{stem}_down{ds}_*.png")
    matches = glob.glob(pat)
    if not matches:
        return None
    try:
        matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    except Exception:
        matches.sort(reverse=True)
    return matches[0]

def _tissue_mask_detect_preview_tag(slide_name: str, downsample: int) -> str | None:
    """
    If a preview exists, extract its color tag from filename:
      <stem>_down<ds>_<tag>.png  -> <tag>
    """
    p = _tissue_mask_any_preview_path(slide_name, downsample)
    if not p:
        return None
    stem = _slide_to_result_stem(slide_name)
    ds = max(1, int(downsample))
    base = os.path.basename(p)
    prefix = f"{stem}_down{ds}_"
    if base.startswith(prefix) and base.lower().endswith(".png"):
        return base[len(prefix):-4]
    return None

def _has_tissue_mask(slide_name: str) -> bool:
    try:
        if os.path.exists(_tissue_mask_path(slide_name)):
            return True
        # If raw mask is missing, allow cached preview to drive availability (default ds=16).
        return _tissue_mask_any_preview_path(slide_name, 16) is not None
    except Exception:
        return False

def _tissue_mask_preview_path(slide_name: str, downsample: int, color_tag: str | None = None) -> str:
    stem = _slide_to_result_stem(slide_name)
    ds = max(1, int(downsample))
    tag = color_tag or TISSUE_MASK_COLOR_TAG
    return os.path.join(TISSUE_MASK_PREVIEW_DIR, f"{stem}_down{ds}_{tag}.png")

def _tissue_seg_full_path(slide_name: str) -> str:
    stem = _slide_to_result_stem(slide_name)
    return os.path.join(TISSUE_SEG_FULL_DIR, f"{stem}.png")

def _has_tissue_seg_full(slide_name: str) -> bool:
    try:
        return os.path.exists(_tissue_seg_full_path(slide_name))
    except Exception:
        return False

def _nuclei_preview_path(slide_name: str, downsample: int) -> str:
    stem = _slide_to_result_stem(slide_name)
    ds = max(1, int(downsample))
    return os.path.join(NUCLEI_SEG_PREVIEW_DIR, f"{stem}_class_down{ds}.png")

def _clear_slide_cache():
    try:
        get_slide.cache_clear()
    except Exception:
        pass

@lru_cache(maxsize=32)
def get_slide(slide_name):
    if not _safe_name(slide_name):
        return None
    path = os.path.join(SLIDE_DIR, slide_name)
    slide = openslide.OpenSlide(path)
    dz = DeepZoomGenerator(slide, tile_size=256, overlap=1, limit_bounds=False)
    return {"slide": slide, "dz": dz, "path": path}

@app.route('/')
def index():
    # No login wall: serve UI directly
    return app.send_static_file('index.html')

@app.route('/slides')
def list_slides():
    files = [f for f in os.listdir(SLIDE_DIR) if f.lower().endswith(ALLOWED_EXT)]
    sorted_files = sorted(files)
    app.logger.debug('list_slides -> %d slides: %s', len(sorted_files), sorted_files[:10])
    return jsonify(sorted_files)

@app.route('/metadata/<path:slide_name>')
def metadata(slide_name):
    app.logger.debug('metadata request: %s', slide_name)
    s = get_slide(slide_name)
    if not s:
        app.logger.debug('metadata: slide not found %s', slide_name)
        return abort(404)
    slide = s['slide']
    props = slide.properties
    mpp_x = props.get('openslide.mpp-x') or props.get('openslide.mpp_x') or props.get('aperio.AppMag')
    # Try to detect objective magnification reported in properties
    obj = props.get('openslide.objective-power') or props.get('aperio.AppMag') or props.get('objective')
    try:
        mpp = float(mpp_x) if mpp_x else None
    except:
        mpp = None
    try:
        objective = float(obj) if obj else None
    except:
        objective = None
    app.logger.debug('metadata: slide=%s width=%s height=%s mpp=%s objective=%s', slide_name, slide.dimensions[0], slide.dimensions[1], mpp, objective)
    # Add pyramid level information from DeepZoomGenerator (level index, dimensions, downsample factor)
    try:
        dz = get_slide(slide_name)['dz']
        levels = []
        full_w = slide.dimensions[0]
        for lvl in range(dz.level_count):
            w, h = dz.get_level_dimensions(lvl)
            ds = max(1, int(round(full_w / w))) if w else 1
            levels.append({ 'level': lvl, 'width': int(w), 'height': int(h), 'downsample': int(ds) })
    except Exception as e:
        app.logger.debug('metadata: failed to compute pyramid levels: %s', e)
        levels = []

    # If only a cached tissue_mask preview exists (no raw mask), align tag with that preview
    # so the front-end request URL stays stable and cache-friendly.
    try:
        detected_tag = _tissue_mask_detect_preview_tag(slide_name, 16)
    except Exception:
        detected_tag = None
    tissue_tag = detected_tag or TISSUE_MASK_COLOR_TAG

    return jsonify({
        "width": slide.dimensions[0],
        "height": slide.dimensions[1],
        "mpp": mpp,
        "objective": objective,
        "overlays": {
            "nuclei_seg": _has_nuclei_seg(slide_name),
            "nuclei_seg_dzi": _has_nuclei_seg_dzi(slide_name),
            "tissue_mask": _has_tissue_mask(slide_name),
            "tissue_seg_full": _has_tissue_seg_full(slide_name),
        },
        "overlay_meta": {
            "tissue_mask_color_tag": tissue_tag,
            "tissue_mask_rgba": list(TISSUE_MASK_RGBA),
        },
        "pyramid": levels
    })


def _parse_int_query(name: str, default: int, min_value: int, max_value: int) -> int:
    v = request.args.get(name, None)
    if v is None:
        return default
    try:
        iv = int(v)
    except Exception:
        return default
    return max(min_value, min(max_value, iv))


@cache.memoize()
def _nuclei_seg_class_preview_png(mask_path: str, downsample: int) -> bytes:
    """
    Build a downsampled RGBA preview (background transparent) for nuclei semantic mask.
    Prefer streaming scanlines via pypng (if installed). Fallback to Pillow if pypng is missing.
    """
    try:
        import png  # from pypng
    except Exception:
        png = None

    # fixed palette for labels 0..5
    # 0 background -> transparent
    palette = {
        0: (0, 0, 0, 0),
        1: (230, 25, 75, 255),     # Neoplastic
        2: (0, 130, 200, 255),     # Inflammatory
        3: (60, 180, 75, 255),     # Connective
        4: (128, 128, 128, 255),   # Dead
        5: (255, 225, 25, 255),    # Epithelial
    }

    ds = max(1, int(downsample))

    if png is not None:
        reader = png.Reader(filename=mask_path)
        w, h, rows, info = reader.read()
        bitdepth = int(info.get('bitdepth', 8))
        if bitdepth != 8:
            raise ValueError(f"Unsupported bitdepth={bitdepth} for nuclei mask PNG")
        if not info.get('greyscale', False):
            raise ValueError("Expected greyscale nuclei mask PNG")

        out_w = (w + ds - 1) // ds
        out_h = (h + ds - 1) // ds

        def out_rows():
            out_y = 0
            for in_y, row in enumerate(rows):
                if in_y % ds != 0:
                    continue
                out = bytearray(out_w * 4)
                for ox in range(out_w):
                    ix = ox * ds
                    if ix >= w:
                        ix = w - 1
                    v = int(row[ix])
                    r, g, b, a = palette.get(v, (255, 255, 255, 255))
                    j = ox * 4
                    out[j] = r
                    out[j + 1] = g
                    out[j + 2] = b
                    out[j + 3] = a
                yield out
                out_y += 1
                if out_y >= out_h:
                    break

        buf = io.BytesIO()
        # greyscale=False -> RGBA (4 channels). Otherwise pypng assumes GA (2 channels).
        png.Writer(out_w, out_h, alpha=True, greyscale=False, bitdepth=8).write(buf, out_rows())
        return buf.getvalue()

    # Pillow fallback
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(mask_path).convert("L")
    w, h = im.size
    out_w = (w + ds - 1) // ds
    out_h = (h + ds - 1) // ds
    small = im.resize((out_w, out_h), resample=Image.NEAREST)
    # Build RGBA via point-maps
    # alpha: 255 where label>0 else 0
    alpha = small.point(lambda p: 255 if p > 0 else 0, mode="L")
    # RGB channels from palette (defaults to white)
    rLUT = [palette.get(i, (255, 255, 255, 255))[0] for i in range(256)]
    gLUT = [palette.get(i, (255, 255, 255, 255))[1] for i in range(256)]
    bLUT = [palette.get(i, (255, 255, 255, 255))[2] for i in range(256)]
    r = small.point(rLUT, mode="L")
    g = small.point(gLUT, mode="L")
    b = small.point(bLUT, mode="L")
    rgba = Image.merge("RGBA", (r, g, b, alpha))
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    return buf.getvalue()


@app.route('/overlay/nuclei_seg/<path:slide_name>.png')
def overlay_nuclei_seg(slide_name):
    """
    Return a downsampled RGBA overlay (PNG) for nuclei semantic segmentation.
    Query:
      - down: integer downsample factor (default 16)
    """
    if not _safe_name(slide_name):
        return abort(404)
    mask_path = _nuclei_class_mask_path(slide_name)
    if not os.path.exists(mask_path):
        return abort(404)

    down = _parse_int_query('down', default=16, min_value=1, max_value=256)
    preview_path = _nuclei_preview_path(slide_name, down)
    # If we've already built a preview, serve it directly (fast).
    try:
        if os.path.exists(preview_path):
            return send_file(preview_path, mimetype='image/png', conditional=True)
    except Exception:
        pass
    try:
        data = _nuclei_seg_class_preview_png(mask_path, down)
        # Persist preview so subsequent loads are instant.
        try:
            os.makedirs(os.path.dirname(preview_path), exist_ok=True)
            tmp = preview_path + ".tmp"
            with open(tmp, 'wb') as f:
                f.write(data)
            os.replace(tmp, preview_path)
        except Exception:
            pass
    except Exception as e:
        app.logger.exception('overlay_nuclei_seg failed slide=%s mask=%s down=%s', slide_name, mask_path, down)
        return jsonify({'ok': False, 'error': str(e)}), 500

    resp = Response(data, mimetype='image/png')
    # cache at client/proxy; server memoizes too
    resp.headers['Cache-Control'] = 'public, max-age=86400'
    return resp


@cache.memoize()
def _tissue_mask_preview_png(mask_path: str, downsample: int, color_tag: str | None = None) -> bytes:
    """
    Build a downsampled RGBA preview for tissue mask.
    - background (0) -> transparent
    - tissue (>0) -> solid color with alpha
    """
    ds = max(1, int(downsample))
    # color for tissue area
    # More visible overlay by default (alpha is combined with front-end opacity).
    # NOTE: color_tag is intentionally part of the memoize key; it can be used to bust cache.
    _ = color_tag
    tr, tg, tb, ta = TISSUE_MASK_RGBA
    try:
        import png  # from pypng
    except Exception:
        png = None

    if png is not None:
        reader = png.Reader(filename=mask_path)
        w, h, rows, info = reader.read()
        bitdepth = int(info.get('bitdepth', 8))
        if bitdepth != 8:
            raise ValueError(f"Unsupported bitdepth={bitdepth} for tissue mask PNG")
        if not info.get('greyscale', False):
            raise ValueError("Expected greyscale tissue mask PNG")

        out_w = (w + ds - 1) // ds
        out_h = (h + ds - 1) // ds

        def out_rows():
            out_y = 0
            for in_y, row in enumerate(rows):
                if in_y % ds != 0:
                    continue
                out = bytearray(out_w * 4)
                for ox in range(out_w):
                    ix = ox * ds
                    if ix >= w:
                        ix = w - 1
                    v = int(row[ix])
                    j = ox * 4
                    if v > 0:
                        out[j] = tr
                        out[j + 1] = tg
                        out[j + 2] = tb
                        out[j + 3] = ta
                    else:
                        out[j] = 0
                        out[j + 1] = 0
                        out[j + 2] = 0
                        out[j + 3] = 0
                yield out
                out_y += 1
                if out_y >= out_h:
                    break

        buf = io.BytesIO()
        # greyscale=False -> RGBA (4 channels). Otherwise pypng assumes GA (2 channels).
        png.Writer(out_w, out_h, alpha=True, greyscale=False, bitdepth=8).write(buf, out_rows())
        return buf.getvalue()

    # Pillow fallback (no extra dependency)
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(mask_path).convert("L")
    w, h = im.size
    out_w = (w + ds - 1) // ds
    out_h = (h + ds - 1) // ds
    small = im.resize((out_w, out_h), resample=Image.NEAREST)
    alpha = small.point(lambda p: ta if p > 0 else 0, mode="L")
    rgba = Image.new("RGBA", (out_w, out_h), (tr, tg, tb, 0))
    rgba.putalpha(alpha)
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    return buf.getvalue()


@app.route('/overlay/tissue_mask/<path:slide_name>.png')
def overlay_tissue_mask(slide_name):
    """
    Return a downsampled RGBA overlay (PNG) for tissue mask.
    Query:
      - down: integer downsample factor (default 16)
    """
    if not _safe_name(slide_name):
        return abort(404)
    down = _parse_int_query('down', default=16, min_value=1, max_value=256)
    # Preferred preview path for current color tag
    preview_path = _tissue_mask_preview_path(slide_name, down, TISSUE_MASK_COLOR_TAG)
    try:
        if os.path.exists(preview_path):
            return send_file(preview_path, mimetype='image/png', conditional=True)
    except Exception:
        pass

    # If raw mask is missing, but an older preview exists (different tag), serve it.
    mask_path = _tissue_mask_path(slide_name)
    if not os.path.exists(mask_path):
        any_preview = _tissue_mask_any_preview_path(slide_name, down)
        if any_preview and os.path.exists(any_preview):
            return send_file(any_preview, mimetype='image/png', conditional=True)
        return abort(404)

    try:
        data = _tissue_mask_preview_png(mask_path, down, TISSUE_MASK_COLOR_TAG)
        try:
            os.makedirs(os.path.dirname(preview_path), exist_ok=True)
            tmp = preview_path + ".tmp"
            with open(tmp, 'wb') as f:
                f.write(data)
            os.replace(tmp, preview_path)
        except Exception:
            pass
    except Exception as e:
        app.logger.exception('overlay_tissue_mask failed slide=%s mask=%s down=%s', slide_name, mask_path, down)
        return jsonify({'ok': False, 'error': str(e)}), 500

    resp = Response(data, mimetype='image/png')
    resp.headers['Cache-Control'] = 'public, max-age=86400'
    return resp


@app.route('/overlay/tissue_seg_full/<path:slide_name>.png')
def overlay_tissue_seg_full(slide_name):
    """
    Return the tissue segmentation visualization (RGB) from result/tissue_seg/full_images.
    This image is typically downsampled (e.g. ~16x) already, so we serve it directly.
    """
    if not _safe_name(slide_name):
        return abort(404)
    p = _tissue_seg_full_path(slide_name)
    if not os.path.exists(p):
        return abort(404)
    resp = send_file(p, mimetype='image/png', conditional=True)
    resp.headers['Cache-Control'] = 'public, max-age=86400'
    return resp


@app.route('/overlay_dzi/nuclei_seg/<path:slide_name>.dzi')
def overlay_nuclei_seg_dzi(slide_name):
    """Serve pre-generated nuclei semantic mask DZI (see tools/make_mask_dzi.py)."""
    if not _safe_name(slide_name):
        return abort(404)
    dzi_path = _nuclei_class_dzi_path(slide_name)
    if os.path.exists(dzi_path):
        return send_file(dzi_path, mimetype='application/xml', conditional=True)
    # Compatibility: if only *_files exists, synthesize a DZI descriptor on the fly.
    tiles_dir = _nuclei_class_dzi_tiles_dir(slide_name)
    if not os.path.exists(tiles_dir):
        return abort(404)
    try:
        s = get_slide(slide_name)
        if not s:
            return abort(404)
        w, h = s['slide'].dimensions
    except Exception:
        return abort(500)
    # Our generator uses dzsave with: tile-size=256, overlap=0, suffix .png
    dzi_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"\n'
        '       Format="png"\n'
        '       Overlap="0"\n'
        '       TileSize="256">\n'
        f'  <Size Width="{int(w)}" Height="{int(h)}"/>\n'
        '</Image>\n'
    )
    resp = Response(dzi_xml, mimetype='application/xml')
    resp.headers['Cache-Control'] = 'public, max-age=86400'
    return resp


@app.route('/overlay_dzi/nuclei_seg/<path:slide_name>_files/<int:level>/<tile>')
def overlay_nuclei_seg_dzi_tile(slide_name, level, tile):
    """Serve pre-generated nuclei semantic mask DZI tiles."""
    if not _safe_name(slide_name):
        return abort(404)
    # Allow serving tiles even if .dzi descriptor file is missing.
    tiles_dir = _nuclei_class_dzi_tiles_dir(slide_name)
    lvl_dir = os.path.join(tiles_dir, str(level))
    if not os.path.exists(lvl_dir):
        return abort(404)
    tile_path = os.path.join(lvl_dir, tile)
    if not os.path.exists(tile_path):
        return abort(404)
    # png tiles
    return send_file(tile_path, mimetype='image/png', conditional=True)

@app.route('/thumbnail/<path:slide_name>.jpg')
def thumbnail(slide_name):
    """Return a small thumbnail (jpeg) for management UI."""
    app.logger.debug('thumbnail request: %s', slide_name)
    if not _safe_name(slide_name):
        app.logger.debug('thumbnail: invalid slide name %s', slide_name)
        return abort(404)
    try:
        s = get_slide(slide_name)['slide']
        # Get thumbnail with max size ~ 256x256
        thumb = s.get_thumbnail((256, 256))
        buf = io.BytesIO()
        thumb.save(buf, format='JPEG', quality=80)
        buf.seek(0)
        return send_file(buf, mimetype='image/jpeg')
    except Exception as e:
        app.logger.exception('thumbnail generation failed for %s', slide_name)
        return abort(500)

@app.route('/upload', methods=['POST'])
def upload():
    """Accept a multipart upload with field 'slide'."""
    app.logger.debug('upload request from %s, content_length=%s', request.remote_addr, request.content_length)
    if 'slide' not in request.files:
        app.logger.debug('upload: no file field present')
        return jsonify({'ok': False, 'error': 'no file'}), 400
    f = request.files['slide']
    if f.filename == '':
        app.logger.debug('upload: empty filename')
        return jsonify({'ok': False, 'error': 'empty filename'}), 400
    filename = secure_filename(f.filename)
    app.logger.debug('upload: filename=%s', filename)
    if not _allowed_file(filename):
        app.logger.debug('upload: file type not allowed %s', filename)
        return jsonify({'ok': False, 'error': 'file type not allowed'}), 400
    dest = os.path.join(SLIDE_DIR, filename)
    if os.path.exists(dest):
        app.logger.debug('upload: file exists %s', dest)
        return jsonify({'ok': False, 'error': 'file exists'}), 400
    try:
        f.save(dest)
        _clear_slide_cache()
        app.logger.info('upload successful: %s', filename)
        return jsonify({'ok': True, 'filename': filename})
    except Exception as e:
        app.logger.exception('upload failed for %s', filename)
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/delete', methods=['POST'])
def delete():
    """Delete a slide by name (form or json param 'slide')."""
    slide = request.form.get('slide') or (request.get_json() or {}).get('slide')
    app.logger.debug('delete request: %s from %s', slide, request.remote_addr)
    if not slide:
        app.logger.debug('delete: no slide specified')
        return jsonify({'ok': False, 'error': 'no slide specified'}), 400
    if not _safe_name(slide):
        app.logger.debug('delete: invalid slide %s', slide)
        return jsonify({'ok': False, 'error': 'invalid slide'}), 400
    try:
        os.remove(os.path.join(SLIDE_DIR, slide))
        _clear_slide_cache()
        app.logger.info('delete successful: %s', slide)
        return jsonify({'ok': True})
    except Exception as e:
        app.logger.exception('delete failed for %s', slide)
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/dzi/<path:slide_name>.dzi')
def dzi(slide_name):
    app.logger.debug('dzi request: %s', slide_name)
    s = get_slide(slide_name)
    if not s:
        app.logger.debug('dzi: slide not found %s', slide_name)
        return abort(404)
    dz = s['dz']
    # Use DeepZoomGenerator.get_dzi to build proper DZI XML
    xml = dz.get_dzi('jpeg')
    return Response(xml, mimetype='application/xml')

# OpenSeadragon expects tiles to be served from a _files/ path next to the .dzi
@app.route('/dzi/<path:slide_name>_files/<int:level>/<tile>')
def dzi_files_tile(slide_name, level, tile):
    """Serve tiles using the DZI _files path (e.g., /dzi/slide.svs_files/16/0_0.jpg)"""
    app.logger.debug('dzi_files_tile request: %s level=%s tile=%s', slide_name, level, tile)
    if not _safe_name(slide_name):
        return abort(404)
    name, ext = os.path.splitext(tile)
    try:
        col, row = map(int, name.split('_'))
    except Exception:
        return abort(404)
    s = get_slide(slide_name)
    if not s:
        return abort(404)
    dz = s['dz']
    try:
        img = dz.get_tile(level, (col, row))
    except Exception:
        return abort(404)
    buf = io.BytesIO()
    fmt = ext.lower().lstrip('.')
    if fmt in ('jpg', 'jpeg'):
        img.save(buf, format='JPEG', quality=85)
        mimetype = 'image/jpeg'
    else:
        img.save(buf, format='PNG')
        mimetype = 'image/png'
    buf.seek(0)
    return send_file(buf, mimetype=mimetype)

@app.route('/tiles/<path:slide_name>/<int:level>/<tile>.jpg')
def tile(slide_name, level, tile):
    app.logger.debug('tile request: slide=%s level=%s tile=%s', slide_name, level, tile)
    s = get_slide(slide_name)
    if not s:
        app.logger.debug('tile: slide not found %s', slide_name)
        return abort(404)
    dz = s['dz']
    try:
        col, row = map(int, tile.split('_'))
    except Exception:
        app.logger.debug('tile: invalid tile name %s', tile)
        return abort(400)
    try:
        img = dz.get_tile(level, (col, row))
    except Exception as e:
        app.logger.exception('tile generation failed: %s', e)
        return abort(404)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    buf.seek(0)
    return send_file(buf, mimetype='image/jpeg')

@app.route('/user')
def user_info():
    """Return current session user info."""
    return jsonify({'user': None})

@app.route('/login', methods=['GET','POST'])
def login():
    """GET serves the login page; POST accepts credentials (JSON) and sets session."""
    if request.method == 'GET':
        # Login disabled; redirect to main UI
        return redirect('/')
    # POST
    data = request.get_json() or request.form or {}
    username = data.get('username')
    password = data.get('password')
    app.logger.debug('login attempt username=%s', username)
    # default demo credential
    if username == 'admin' and password == '1234':
        session['user'] = username
        app.logger.info('login successful for %s', username)
        return jsonify({'ok': True, 'user': username})
    app.logger.debug('login failed for %s', username)
    return jsonify({'ok': False, 'error': 'invalid credentials'}), 401

@app.route('/logout', methods=['POST'])
def logout():
    user = session.pop('user', None)
    app.logger.info('logout (no-op): %s', user)
    return jsonify({'ok': True})

@app.route('/debug')
def debug_info():
    """Simple diagnostic endpoint for debugging client-server interactions."""
    files = [f for f in os.listdir(SLIDE_DIR) if f.lower().endswith(ALLOWED_EXT)]
    return jsonify({'slides_dir': SLIDE_DIR, 'slide_count': len(files), 'slides': files[:50]})

if __name__ == '__main__':
    # Ensure slides dir exists and print diagnostic info
    if not os.path.exists(SLIDE_DIR):
        print(f"Slides directory '{SLIDE_DIR}' does not exist. Creating it.")
        os.makedirs(SLIDE_DIR, exist_ok=True)
    else:
        files = [f for f in os.listdir(SLIDE_DIR) if f.lower().endswith(ALLOWED_EXT)]
        print(f"Slides directory: {SLIDE_DIR} - {len(files)} slides found")
        if files:
            for f in files[:20]:
                print("  -", f)
    # Ensure cache dir exists (filesystem cache)
    try:
        cache_dir = app.config.get('CACHE_DIR')
        if Cache and cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Cache directory created: {cache_dir}")
    except Exception:
        pass
    # Print result/overlay diagnostics
    try:
        print(f"Result directory: {RESULT_DIR}")
        print(f"Nuclei seg directory: {NUCLEI_SEG_DIR} (exists={os.path.exists(NUCLEI_SEG_DIR)})")
        print(f"Nuclei seg preview dir: {NUCLEI_SEG_PREVIEW_DIR} (exists={os.path.exists(NUCLEI_SEG_PREVIEW_DIR)})")
    except Exception:
        pass
    # Enable debug logging to see tile requests
    try:
        import logging
        app.logger.setLevel(logging.DEBUG)
    except Exception:
        pass
    app.run(host='0.0.0.0', port=8366, debug=True)
