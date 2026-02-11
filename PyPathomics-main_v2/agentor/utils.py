import os.path
import openslide
from pathlib import Path


def get_swi_thumbnail(swi_path, img_size=(256, 256)):
    file_path = Path(swi_path)
    img_tmp = os.path.join(file_path.parent.parent, 'tmp')
    os.makedirs(img_tmp, exist_ok=True)
    img_path = os.path.join(img_tmp, file_path.name.replace(file_path.suffix, '.png'))
    if os.path.exists(img_path):
        return img_path
    swi_slide = openslide.OpenSlide(file_path)
    high_lev = swi_slide.level_dimensions[-1]
    img = swi_slide.read_region((0, 0), swi_slide.level_count - 1, high_lev)
    img = img.convert('RGB')
    img = img.resize(img_size)
    img.save(img_path)
    return img_path


def generate_report():
    """"""
