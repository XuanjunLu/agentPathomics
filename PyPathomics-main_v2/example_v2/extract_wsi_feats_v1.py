#!/usr/bin/env python

from __future__ import print_function

import logging

import pathomics
from pathomics import featureextractor
import pandas as pd

# import multiprocessing
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
# from pathos.multiprocessing import ProcessingPool as Pool
from itertools import repeat
from functools import partial
import numpy as np
from pathlib import Path
from PIL import Image
from skimage import measure
Image.MAX_IMAGE_PIXELS=None
from pathomics.helper import get_file_handler, OpenSlideHandler

# Get the Pypathomics logger (default log-level = INFO)
# logger = pathomics.logger
# logger.setLevel(
#     logging.DEBUG
# )  # set level to DEBUG to include debug log messages in log file

# # Set up the handler to write out all log entries to a file
# handler = logging.FileHandler(filename='wsi_log.txt', mode='w')
# formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
# handler.setFormatter(formatter)
# logger.addHandler(handler)

def read_patch_from_wsi(wsi_obj, coords, patch_size, save_path):
    x_tile, y_tile = coords
    w_tile, h_tile = patch_size,patch_size
    patch_tile = wsi_obj.read_region((x_tile,y_tile), (w_tile,h_tile))
    if isinstance(patch_tile, np.ndarray):
        patch_tile = Image.fromarray(patch_tile)
    if save_path.exists():
        return save_path
    patch_tile.save(save_path)
    return save_path

def read_patch_from_np(np_img, coords, patch_size, save_path, downscale=1):
    x_tile, y_tile = coords
    x_tile, y_tile = x_tile//downscale, y_tile//downscale
    w_tile, h_tile = patch_size//downscale, patch_size//downscale
    np_tile = np_img[y_tile:y_tile+h_tile, x_tile:x_tile+w_tile]
    if downscale != 1:
        np_tile = Image.fromarray(np_tile)
        np_tile = np_tile.resize((patch_size, patch_size), Image.NEAREST)
    if isinstance(np_tile, np.ndarray):
        np_tile = Image.fromarray(np_tile)
    if save_path.exists():
        return save_path
    np_tile.save(save_path)
    return save_path

def extract_wsi_patches(wsi_file, coords, patch_size, save_dir, suffix, mag_info=40, n_workers=1):
    wsi_obj = get_file_handler(wsi_file)
    wsi_obj.prepare_reading(read_mag=mag_info)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    wsi_name = wsi_file.stem
    save_paths = [save_dir / f"{wsi_name}_{x}_{y}{suffix}" for x,y in coords]
    
    with ThreadPool(n_workers) as pool:
        pool.starmap(
            read_patch_from_wsi, 
            zip(repeat(wsi_obj), coords, repeat(patch_size), save_paths)
        )
    return save_paths

def extract_mask_patches(image, image_name, coords, patch_size, save_dir, suffix, scale=1, n_workers=1):
    
    save_dir.mkdir(parents=True, exist_ok=True)
    save_paths = [save_dir / f"{image_name}_{x}_{y}{suffix}" for x,y in coords]

    with ThreadPool(n_workers) as pool:
        pool.starmap(
            read_patch_from_np, 
            zip(repeat(image), coords, repeat(patch_size), save_paths, repeat(scale))
        )
    
    return save_paths
        
def gen_coords(width, height, patch_size, stride):
    coords = []
    x = 0
    y = 0
    stride = patch_size if stride is None else stride
    x_stride = y_stride = stride
    tmp_x = x
    tmp_y = y

    while True:
        if tmp_y == y + height:
            break
        if tmp_y + y_stride > y + height:
            tmp_y = tmp_y - (y_stride - (y + height - tmp_y))
        coords.append((tmp_x, tmp_y))
        tmp_x += x_stride
        if tmp_x == x + width:
            tmp_x = x
            tmp_y += y_stride
        elif tmp_x + x_stride > x + width:
            tmp_x = tmp_x - (x_stride - (x + width - tmp_x))
            if tmp_x == x:
                tmp_x = x
                tmp_y += y_stride
    return coords

def check_coords_in_mask(coord, patch_size, dimension, masks):
    Width, Height = dimension # wsi width and height
    
    for mask in masks:
        h,w = mask.shape
        scale = Width // w
        patch_mask = mask[coord[1]//scale:coord[1]//scale+patch_size//scale, coord[0]//scale:coord[0]//scale+patch_size//scale]
        if patch_mask.sum() == 0:
            return False
    return True

def extract_nuclei_feats(wsi_name, image, nuclei_instance_mask, nuclei_class_mask, tumor_mask, save_dir, coord=None):
    # Initialize feature extractor
    nuclei_extractor = featureextractor.PathomicsFeatureExtractor()
    nuclei_extractor.enableFeaturesByName(
        firstorder=[],
        glcm=[],
        # glrlm=[], skil this, too slow
        # glszm=[],
        # ngtdm=[],
        # gldm=[],
        contourbased=[],
        basic=[],
    )

    extractor = nuclei_extractor

    save_path = save_dir / f"{wsi_name}_{coord[0]}_{coord[1]}_nuclei_feats.csv"
    if save_path.exists():
        df = pd.read_csv(save_path)
        return df
    image = np.array(Image.open(image))
    nuclei_instance_mask = np.array(Image.open(nuclei_instance_mask))
    nuclei_class_mask = np.array(Image.open(nuclei_class_mask))
    tumor_mask = np.array(Image.open(tumor_mask))
    feats = extractor.execute(image, nuclei_instance_mask)
    wsi_id = wsi_name
    image_id = f"{wsi_id}_{coord[0]}_{coord[1]}"
    feature_names = list(feats.keys())
    columns = feature_names
    df = pd.DataFrame(feats.values()).transpose()
    df.columns = columns
    df.insert(0, 'wsi_id', wsi_id)
    df.insert(1, 'image_id', image_id)
    df.insert(2, 'nuclei_id', [f"{image_id}_{j}" for j in range(len(df))])

    tumor_status = []
    nuclei_class = []
    
    for j in range(len(df)):
        centroid_x = df.loc[j, 'Basic_CentroidX']
        centroid_y = df.loc[j, 'Basic_CentroidY']
        if tumor_mask[int(centroid_y), int(centroid_x)] > 0: # bug
            tumor_status.append(1)
        else:
            tumor_status.append(0)
        nuclei_class.append(nuclei_class_mask[int(centroid_y), int(centroid_x)])
    df['tumor_status'] = tumor_status
    df['nuclei_class'] = nuclei_class
    df.to_csv(save_path, index=False)
    return df

def extract_image_feats(wsi_name, image, nuclei_mask, save_dir, coord=None):
    extractor = featureextractor.PathomicsFeatureExtractor()
    extractor.enableFeaturesByName(
        rgt=[],
        localgraph=[],
        globalgraph=[],
        # spatil=[], skip this, need lymphocytes masks
        flock=[],
    )
    save_path = save_dir / f"{wsi_name}_{coord[0]}_{coord[1]}_image_feats.csv"
    if save_path.exists():
        df = pd.read_csv(save_path)
        return df
    try:
        image = np.array(Image.open(image))
        nuclei_mask = np.array(Image.open(nuclei_mask))
        feats = extractor.execute(image, nuclei_mask)
        wsi_id = wsi_name
        image_id = f"{wsi_id}_{coord[0]}_{coord[1]}"
        feature_names = list(feats.keys())
        columns = feature_names
        df = pd.DataFrame(feats.values()).transpose()
        df.columns = columns
        df.insert(0, 'wsi_id', wsi_id)
        df.insert(1, 'image_id', image_id)
        df.to_csv(save_path, index=False)
    except Exception as e:
        print(wsi_name, coord, e)
        # logging error
        # logger.error(f"Error in extracting image features for {wsi_name} at {coord}: {e}")
        return None
    return df

def read_mask(mask_path):
    mask = Image.open(mask_path)
    return np.array(mask)


def extract_feats_from_wsi(wsi_path:Path, nuclei_instance_path:Path, nuclei_class_path:Path, tumor_path:Path, save_dir:Path, mag: int, patch_size=1024, n_workers=32):

    wsi_name = wsi_path.stem
    nuclei_save_path = save_dir / "{}_nuclei_feats.csv".format(wsi_name)
    image_save_path = save_dir / "{}_image_feats.csv".format(wsi_name)
    
    # return results if already processed
    if nuclei_save_path.exists() and image_save_path.exists():
        return pd.read_csv(nuclei_save_path), pd.read_csv(image_save_path)
    
    wsi_obj = get_file_handler(wsi_path)
    width, height = wsi_obj.get_dimensions(read_mag = mag)
    tumor_mask = read_mask(tumor_path)
    nuclei_instance_mask = read_mask(nuclei_instance_path)
    nuclei_class_mask = read_mask(nuclei_class_path)
    coords = gen_coords(width, height, patch_size, patch_size)
    # only select coords within nuclei and tumor masks
    coords = [
        coord for coord in coords if check_coords_in_mask(coord, patch_size, (width, height), [nuclei_instance_mask,tumor_mask])
    ]
    # coords = coords[:50]

    if len(coords) == 0:
        print(f"No valid coords for {wsi_name}")
        return None, None

    save_dir.mkdir(parents=True, exist_ok=True)

    image_patches = extract_wsi_patches(
                            wsi_file=wsi_path,
                            coords=coords,
                            patch_size=patch_size,
                            save_dir=save_dir/wsi_name,
                            suffix='.png',
                            n_workers=n_workers,
                            mag_info=40,
    )
    nuclei_instance_patches = extract_mask_patches(
                            image=nuclei_instance_mask,
                            image_name=wsi_name,
                            coords=coords,
                            patch_size=patch_size,
                            save_dir=save_dir/wsi_name,
                            suffix='_nuclei_instance_mask.png',
                            n_workers=n_workers,
                            scale=width//nuclei_instance_mask.shape[1],
    )
    nuclei_class_patches = extract_mask_patches(
                            image=nuclei_class_mask,
                            image_name=wsi_name,
                            coords=coords,
                            patch_size=patch_size,
                            save_dir=save_dir/wsi_name,
                            suffix='_nuclei_class_mask.png',
                            n_workers=n_workers,
                            scale=width//nuclei_class_mask.shape[1],
    )
    tumor_patches = extract_mask_patches(
                            image=tumor_mask,
                            image_name=wsi_name,
                            coords=coords,
                            patch_size=patch_size,
                            save_dir=save_dir/wsi_name,
                            suffix='_tumor_mask.png',
                            n_workers=n_workers,
                            scale=width//tumor_mask.shape[1],
    )

    # release or delete tumor_mask and nuclei mask
    del tumor_mask, nuclei_instance_mask, nuclei_class_mask

    #select coords for image_feats and also for nuclei featus, with min size nuclei in a patch, comment this if need all patches for nuclei level
    # min_nuclei_size = 50
    # indices_to_remove = []

    # for i, patch in enumerate(nuclei_patches):
    #     nuclei_mask = read_mask(patch)
    #     num_nuclei  = len(np.unique(measure.label(nuclei_mask))) - 1
    #     if num_nuclei < min_nuclei_size:
    #         indices_to_remove.append(i)

    # # Remove the elements from the lists after the loop
    # for idx in reversed(indices_to_remove):
    #     coords.pop(idx)
    #     image_patches.pop(idx)
    #     nuclei_patches.pop(idx)
    #     tumor_patches.pop(idx)

    # extract features
    with Pool(n_workers) as pool:
        nuclei_feats = pool.starmap(
            extract_nuclei_feats, 
            zip(
                repeat(wsi_name), image_patches, nuclei_instance_patches, nuclei_class_patches, tumor_patches, repeat(save_dir / wsi_name), coords
            )
        )
    
    #select coords for image_feats, with min size nuclei in a patch
    min_nuclei_number = 50
    indices_to_remove = []

    for i, coord in enumerate(coords):
        nuclei_feat = nuclei_feats[i]
        if len(nuclei_feat) < min_nuclei_number:
            indices_to_remove.append(i)

    # Remove the elements from the lists after the loop
    for idx in reversed(indices_to_remove):
        coords.pop(idx)
        image_patches.pop(idx)
        nuclei_instance_patches.pop(idx)
        nuclei_class_patches.pop(idx)
        tumor_patches.pop(idx)


    with Pool(n_workers) as pool:
        image_feats = pool.starmap(
            extract_image_feats,
            zip(
                repeat(wsi_name), image_patches, nuclei_instance_patches, repeat(save_dir / wsi_name), coords
            )
        )
    # image_feats = []
    # for i, (image_patch, nuclei_patch) in enumerate(zip(image_patches, nuclei_patches)):
    #     image_feat = extract_image_feats(wsi_name, image_patch, nuclei_patch, save_dir / wsi_name, coords[i])
    #     image_feats.append(image_feat)
        
    image_feats = [feat for feat in image_feats if feat is not None]

    nuclei_feats = pd.concat(nuclei_feats).reset_index(drop=True)
    image_feats = pd.concat(image_feats).reset_index(drop=True)
    nuclei_feats.to_csv(nuclei_save_path, index=False)
    image_feats.to_csv(image_save_path, index=False)
    return nuclei_feats, image_feats
    
def main():
    wsi_dir = Path("/mnt/hd0/project/bca21gene/data/fahzu/svs")
    nuclei_dir = Path("/mnt/hd0/project/bca21gene/generated/fahzu/nuclei_seg/tiff2")
    tumor_dir = Path('/mnt/hd0/project/bca21gene/generated/fahzu/tumor_seg')
    save_dir = Path("/mnt/hd0/project/bca21gene/generated/fahzu/feats")
    # import shutil
    # wsi_dir = Path('/mnt/hd0/project/pypathomics/exmaple_data/svs')
    # nuclei_dir = Path('/mnt/hd0/project/pypathomics/exmaple_data/nuclei_seg')
    # tumor_dir = Path('/mnt/hd0/project/pypathomics/exmaple_data/tumor_seg')
    # save_dir = Path('/mnt/hd0/project/pypathomics/exmaple_data/feats')
    # shutil.rmtree(save_dir, ignore_errors=True)
    # save_dir.mkdir(parents=True, exist_ok=True)

    wsi_paths = list(wsi_dir.glob("*.*"))
    for wsi_path in wsi_paths:
        print(wsi_path.stem)
        nuclei_instance_path = nuclei_dir / f"{wsi_path.stem}_instance.tiff"
        nuclei_class_path = nuclei_dir / f"{wsi_path.stem}_class.tiff"
        tumor_path = tumor_dir / f"{wsi_path.stem}.png"
        extract_feats_from_wsi(wsi_path, nuclei_instance_path, nuclei_class_path, tumor_path, save_dir, mag=40, patch_size=1024, n_workers=8)

if __name__ == '__main__':
    main()

