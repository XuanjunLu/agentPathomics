#!/usr/bin/env python

from __future__ import print_function

import logging
import uuid

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
from tqdm import tqdm

# Get the Pypathomics logger (default log-level = INFO)
logger = pathomics.logger
logger.setLevel(
    logging.INFO
)  # set level to DEBUG to include debug log messages in log file

# # Set up the handler to write out all log entries to a file
handler = logging.FileHandler(filename='wsi_log.txt', mode='w')
formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def read_patch_from_wsi(wsi_obj, coords, patch_size, save_path):
    if save_path.exists():
        return save_path
    x_tile, y_tile = coords
    w_tile, h_tile = patch_size, patch_size
    patch_tile = wsi_obj.read_region((x_tile, y_tile), (w_tile, h_tile))
    if isinstance(patch_tile, np.ndarray):
        patch_tile = Image.fromarray(patch_tile)
    patch_tile.save(save_path)
    return save_path

def read_patch_from_np(np_img, coords, patch_size, save_path, downscale=1):
    if save_path.exists():
        return save_path
    x_tile, y_tile = coords
    x_tile, y_tile = x_tile//downscale, y_tile//downscale
    w_tile, h_tile = patch_size//downscale, patch_size//downscale
    np_tile = np_img[y_tile:y_tile+h_tile, x_tile:x_tile+w_tile]
    if downscale != 1:
        np_tile = Image.fromarray(np_tile)
        np_tile = np_tile.resize((patch_size, patch_size), Image.NEAREST)
    if isinstance(np_tile, np.ndarray):
        np_tile = Image.fromarray(np_tile) 
    np_tile.save(save_path)
    return save_path

def extract_wsi_patches(wsi_file, coords, patch_size, save_dir, suffix, mag_info=40, n_workers=1):
    wsi_obj = get_file_handler(wsi_file)
    wsi_obj.prepare_reading(read_mag=mag_info)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    wsi_name = wsi_file.stem
    save_paths = [save_dir / f"{wsi_name}_{x}_{y}{suffix}" for x, y in coords]
    
    with ThreadPool(n_workers) as pool:
        pool.starmap(
            read_patch_from_wsi, 
            zip(repeat(wsi_obj), coords, repeat(patch_size), save_paths)
        )
    return save_paths

def extract_mask_patches(image, image_name, coords, patch_size, save_dir, suffix, scale=1, n_workers=1):
    
    save_dir.mkdir(parents=True, exist_ok=True)
    save_paths = [save_dir / f"{image_name}_{x}_{y}{suffix}" for x, y in coords]

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
    Width, Height = dimension     # wsi width and height
    
    for mask in masks:
        h, w = mask.shape
        scale = Width // w
        patch_mask = mask[coord[1]//scale:coord[1]//scale+patch_size//scale, coord[0]//scale:coord[0]//scale+patch_size//scale]
        if patch_mask.sum() == 0:
            return False
    return True

def extract_patch_feats(image_path, nuclei_instance_path, nuclei_class_path, tumor_path, save_dir, feature_type,
                        nuclei_type=None) -> None:
    """
    Extracting the specified features of the nuclei of the specified type

    :param image_path: str,
    :param nuclei_instance_path: str,
    :param nuclei_class_path: str,
    :param tumor_path: str,
    :param save_dir: str,
    :param feature_type: str, "shape", "texture", "topology", "interplay" or "all", if "all", extract all features
    :param nuclei_type: integer, [1, 2, 3, 4, 5], nuclei type for  feature extraction, if None, get all nuclei

    :return:
    """
    image_feature_names = [
        'RGT',
        'LocalGraph',
        'GlobalGraph',
        'FLocK',
    ]

    nuclei_feature_names = [
        'FirstOrder',
        'GLCM',
        'ContourBased',
        'Basic',
    ]

    shape_features = dict(basic=[], contourbased=[])
    texture_features = dict(basic=[], firstorder=[], glcm=[], glrlm=[], glszm=[], ngtdm=[], gldm=[])
    topology_features = dict(rgt=[],  localgraph=[], globalgraph=[])
    interplay_features = dict(flock=[])
    all_features = dict(basic=[], contourbased=[], firstorder=[], glcm=[], glrlm=[], glszm=[], ngtdm=[], gldm=[],
                        rgt=[],  localgraph=[], globalgraph=[], flock=[])

    extractor = featureextractor.PathomicsFeatureExtractor()
    # extractor.enableFeaturesByName(
    #     # image_levels
    #     rgt=[],
    #     localgraph=[],
    #     globalgraph=[],
    #     # spatil=[], skip this, need lymphocytes masks
    #     flock=[],
    #     # nuclei_levels
    #     firstorder=[],
    #     glcm=[],
    #     # glrlm=[], skil this, too slow
    #     # glszm=[],
    #     # ngtdm=[],
    #     # gldm=[],
    #     contourbased=[],
    #     basic=[],
    # )
    if feature_type == "shape":
        extractor.enableFeaturesByName(**shape_features)
    elif feature_type == "texture":
        extractor.enableFeaturesByName(**texture_features)
    elif feature_type == "topology":
        extractor.enableFeaturesByName(**topology_features)
    elif feature_type == "interplay":
        extractor.enableFeaturesByName(**interplay_features)
    elif feature_type == "all":
        extractor.enableFeaturesByName(**all_features)
    else:
        raise TypeError('Please enter a valid feature type: "shape", "texture", "topology", "interplay" or "all".')


    name = image_path.stem
    wsi_id = name.split('_')[0]
    image_id = name

    if name != nuclei_instance_path.stem or name != nuclei_class_path.stem or name != tumor_path.stem:
        logger.error(f"Error in Patch Extraction: {name} not match")
        exit()

    image_feats_save_path = save_dir / 'image_level' /wsi_id / f"{name}.csv"
    nuclei_feats_save_path = save_dir / 'nuclei_level' /wsi_id / f"{name}.csv"
    
    if image_feats_save_path.exists() and nuclei_feats_save_path.exists():
        return

    image = np.array(Image.open(image_path))
    nuclei_mask = np.array(Image.open(nuclei_instance_path))
    nuclei_class_mask = np.array(Image.open(nuclei_class_path))
    tumor_mask = np.array(Image.open(tumor_path))

    if nuclei_type is not None:
        labels = measure.label(nuclei_mask)
        props = measure.regionprops(labels, image)
        for prop in props:
            centroid_y, centroid_x = prop.centroid
            coords = prop.coords
            y_coords = coords[:, 0]
            x_coords = coords[:, 1]
            if (nuclei_class_mask[int(centroid_y), int(centroid_x)] != nuclei_type) or (prop.area < 20):
                nuclei_mask[y_coords, x_coords] = 0

    min_nuclei_size = 30

    num_nuclei = len(np.unique(measure.label(nuclei_mask))) - 1
    if num_nuclei < min_nuclei_size:
        return

    try:
        feats = extractor.execute(image, nuclei_mask)
    except:
        return

    image_feats = {}
    nuclei_feats = {}
    for k, v in feats.items():
        if k.split('_')[0] in image_feature_names:
            image_feats[k] = v
        elif k.split('_')[0] in nuclei_feature_names:
            nuclei_feats[k] = v
            
    # for image level features
    image_feats_save_path.parent.mkdir(parents=True, exist_ok=True)
    image_level_feature_names = list(image_feats.keys())
    df_image = pd.DataFrame(image_feats.values()).transpose()
    df_image.columns = image_level_feature_names
    df_image.insert(0, 'wsi_id', wsi_id)
    df_image.insert(1, 'image_id', image_id)
    df_image.to_csv(image_feats_save_path, index=False)

    # for nuclei level features
    nuclei_feats_save_path.parent.mkdir(parents=True, exist_ok=True)

    nuclei_level_feature_names = list(nuclei_feats.keys())
    df_nuclei = pd.DataFrame(nuclei_feats.values()).transpose()
    df_nuclei.columns = nuclei_level_feature_names
    df_nuclei.insert(0, 'wsi_id', wsi_id)
    df_nuclei.insert(1, 'image_id', image_id)
    df_nuclei.insert(2, 'nuclei_id', [f"{image_id}_{j}" for j in range(len(df_nuclei))])

    tumor_status = []
    nuclei_class = []
    
    for j in range(len(df_nuclei)):
        centroid_x = df_nuclei.loc[j, 'Basic_CentroidX']
        centroid_y = df_nuclei.loc[j, 'Basic_CentroidY']
        if tumor_mask[int(centroid_y), int(centroid_x)] > 0:   # bug
            tumor_status.append(1)
        else:
            tumor_status.append(0)
        nuclei_class.append(nuclei_class_mask[int(centroid_y), int(centroid_x)])
    df_nuclei['tumor_status'] = tumor_status
    df_nuclei['nuclei_class'] = nuclei_class
    df_nuclei.to_csv(nuclei_feats_save_path, index=False)
    return


def tissue_extract_patch_feats(image_path, roi_path, save_dir):
    extractor = featureextractor.PathomicsFeatureExtractor()
    extractor.enableFeaturesByName(
        # image_levels
        firstorder=[],
        glcm=[],
        # glrlm=[], skil this, too slow
        # glszm=[],
        # ngtdm=[],
        # gldm=[]
    )

    name = image_path.stem
    wsi_id = name.split('_')[0]
    image_id = name

    if name != roi_path.stem:
        logger.error(f"Error in Patch Extraction: {name} not match")
        exit()

    image_feats_save_path = save_dir / 'image_level' / wsi_id / f"{name}.csv"

    if image_feats_save_path.exists():
        return

    image = np.array(Image.open(image_path))
    roi_mask = np.array(Image.open(roi_path))
    h_roi_mask, w_roi_mask = roi_mask.shape[:2]

    min_tissue_ratio = 0.6

    tissue_ratio = np.sum((roi_mask > 1).astype(np.uint8)) / (h_roi_mask * w_roi_mask)

    if tissue_ratio < min_tissue_ratio:
        return
    try:
        feats = extractor.execute(image, roi_mask)
    except:
        return

    image_feats = {}
    for k, v in feats.items():
        image_feats[k] = v

    # for image level features
    image_feats_save_path.parent.mkdir(parents=True, exist_ok=True)
    image_level_feature_names = list(image_feats.keys())
    df_image = pd.DataFrame(image_feats.values()).transpose()
    df_image.columns = image_level_feature_names
    df_image.insert(0, 'wsi_id', wsi_id)
    df_image.insert(1, 'image_id', image_id)
    df_image.to_csv(image_feats_save_path, index=False)

    return


def read_mask(mask_path):
    mask = Image.open(mask_path)
    if len(mask.getbands()) > 1:
        warnings.warn("Please ensure that the mask is a single-channel grayscale image!")
        mask.convert("L")
    return np.array(mask)

def extract_patches(wsi_path, nuclei_instance_path, nuclei_class_path, tumor_path, save_dir, mag=40, patch_size=1024, n_workers=32):
    wsi_obj = get_file_handler(wsi_path)
    wsi_name = wsi_path.stem
    width, height = wsi_obj.get_dimensions(read_mag=mag)
    tumor_mask = read_mask(tumor_path)
    nuclei_instance_mask = read_mask(nuclei_instance_path)
    nuclei_class_mask = read_mask(nuclei_class_path)
    gray_level = len(np.unique(np.array(nuclei_class_path)))
    if gray_level >= 20:
        raise ValueError(f"The grayscale level in {nuclei_class_path} exceeds the limit. "
                         f"Please ensure that the grayscale of the nuclear area corresponds to its respective "
                         f"category number. For example, grayscale level 0 for background, 1 for neoplastic cells, "
                         f"2 for inflammatory cells, 3 for connective tissue cells, 4 for necrotic cells, "
                         f"5 for non-neoplastic cells, 6 for epithelial cells and so on")
    coords = gen_coords(width, height, patch_size, patch_size)
    # only select coords within nuclei and tumor masks
    coords = [
        coord for coord in coords if check_coords_in_mask(coord, patch_size, (width, height),
                                                          [nuclei_instance_mask, tumor_mask])
    ]

    if len(coords) == 0:
        print(f"No valid coords for {wsi_name}")
        logger.error(f"Error in Patch Extraction: No valid coords for {wsi_name}")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    image_patches = extract_wsi_patches(
                            wsi_file=wsi_path,
                            coords=coords,
                            patch_size=patch_size,
                            save_dir=save_dir / 'patches' / 'image' / wsi_name,
                            suffix='.png',
                            n_workers=n_workers,
                            mag_info=40,
    )
    nuclei_instance_patches = extract_mask_patches(
                            image=nuclei_instance_mask,
                            image_name=wsi_name,
                            coords=coords,
                            patch_size=patch_size,
                            save_dir=save_dir / 'patches' / 'nuclei_instance' / wsi_name,
                            suffix='.png',
                            n_workers=n_workers,
                            scale=width//nuclei_instance_mask.shape[1],
    )
    nuclei_class_patches = extract_mask_patches(
                            image=nuclei_class_mask,
                            image_name=wsi_name,
                            coords=coords,
                            patch_size=patch_size,
                            save_dir=save_dir / 'patches' / 'nuclei_class' / wsi_name,
                            suffix='.png',
                            n_workers=n_workers,
                            scale=width//nuclei_class_mask.shape[1],
    )
    tumor_patches = extract_mask_patches(
                            image=tumor_mask,
                            image_name=wsi_name,
                            coords=coords,
                            patch_size=patch_size,
                            save_dir=save_dir / 'patches' / 'tumor' / wsi_name,
                            suffix='.png',
                            n_workers=n_workers,
                            scale=width//tumor_mask.shape[1],
    )
    return


def tissue_extract_patches(wsi_path, roi_path, save_dir, mag=40, patch_size=1024, n_workers=32):
    wsi_obj = get_file_handler(wsi_path)
    wsi_name = wsi_path.stem
    width, height = wsi_obj.get_dimensions(read_mag=mag)
    roi_mask = read_mask(roi_path)
    coords = gen_coords(width, height, patch_size, patch_size)
    # only select coords within nuclei and tumor masks
    coords = [
        coord for coord in coords if check_coords_in_mask(coord, patch_size, (width, height),
                                                          [roi_mask])
    ]

    if len(coords) == 0:
        print(f"No valid coords for {wsi_name}")
        logger.error(f"Error in Patch Extraction: No valid coords for {wsi_name}")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    image_patches = extract_wsi_patches(
                            wsi_file=wsi_path,
                            coords=coords,
                            patch_size=patch_size,
                            save_dir=save_dir / 'patches' / 'image' / wsi_name,
                            suffix='.png',
                            n_workers=n_workers,
                            mag_info=40,
    )
    roi_patches = extract_mask_patches(
                            image=roi_mask,
                            image_name=wsi_name,
                            coords=coords,
                            patch_size=patch_size,
                            save_dir=save_dir / 'patches' / 'roi' / wsi_name,
                            suffix='.png',
                            n_workers=n_workers,
                            scale=width//roi_mask.shape[1],
    )
    return


def merge_wsi_feats(folder, save_path):
    if save_path.exists():
        return
    csv_files = list(folder.glob("*.csv"))
    if len(csv_files) == 0:
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    dfs = [pd.read_csv(csv_file) for csv_file in csv_files]
    df = pd.concat(dfs).reset_index(drop=True)
    df.to_csv(save_path, index=False)
    return
    
def nuclei_feats_extraction(wsi_dir, nuclei_dir, tumor_dir, save_dir, mask_suffix="png", wsi_num=None,
                            mag=40, n_workers=25, feature_type="shape", nuclei_type=None):
    """
    Nuclei-level feature extraction, including "shape", "texture", "topology", and "interplay".

    :param wsi_dir: str, the directory of the folder where WSIs are stored
    :param nuclei_dir: str, the directory of nuclear instance mask (grayscale image, filename contains "instance") and
    classification mask (grayscale image, filename contains "class")
    :param tumor_dir:  str, grayscale image，the directory of tissue (tumor) mask, ensuring that the nuclei are inside
    the tissue
    :param save_dir: str, the directory of saving results
    :param mask_suffix: str, the suffix of nuclear instance mask, classification mask and tissue (tumor) mask
    :param wsi_num: integer, [0, +inf), the selected number of WSIs for feature extraction (e.g., 200)
    :param mag: integer, target magnification for feature extraction (e.g., 40, 20, 10)
    :param n_workers: integer, the number of processes
    :param feature_type: str, "shape", "texture", "topology", "interplay" or "all", if "all", extract all features
    :param nuclei_type: integer, [1, 2, 3, 4, 5]. nuclei type for  feature extraction, if None, get all nuclei

    :return: csv file, feature matrix, each column represent a feature name. In "shape" and "texture", each row
    represents a nucleus, in "topology" and "interplay", each row represents a patch
    """

    wsi_dir = Path(wsi_dir)
    nuclei_dir = Path(nuclei_dir)
    tumor_dir = Path(tumor_dir)
    save_dir = Path(save_dir)

    wsi_paths = list(wsi_dir.glob("*.*"))
    # extracted_list_path = '/mnt/hd0/project/bca21gene/generated/fahzu/extracted_names.csv'
    # extracted_names = pd.read_csv(extracted_list_path)['name'].values

    # wsi_paths = [wsi_path for wsi_path in wsi_paths if wsi_path.stem not in extracted_names]
    # select index%2==0 for this pc
    # wsi_paths = wsi_paths[::2]
    if wsi_num:
        wsi_paths = wsi_paths[:wsi_num]
    else:
        wsi_paths = wsi_paths[:]

    # tqdm showing wsi name and progress, patch extraction
    # for wsi_path in tqdm(wsi_paths, desc='Extracting Patches'):
    #     wsi_name = wsi_path.stem
    #     nuclei_instance_path = nuclei_dir / f"{wsi_name}_instance.{mask_suffix}"
    #     if not nuclei_instance_path.exists():
    #         raise FileNotFoundError(f"{nuclei_instance_path} does not exist!")
    #     nuclei_class_path = nuclei_dir / f"{wsi_name}_class.{mask_suffix}"
    #     if not nuclei_class_path.exists():
    #         raise FileNotFoundError(f"{nuclei_class_path} does not exist!")
    #     tumor_path = tumor_dir / f"{wsi_name}.{mask_suffix}"
    #     if not tumor_path.exists():
    #         raise FileNotFoundError(f"{tumor_path} does not exist!")
    #     extract_patches(wsi_path, nuclei_instance_path, nuclei_class_path, tumor_path, save_dir, mag=mag, n_workers=n_workers)

    image_patch_dir = save_dir / 'patches' / 'image'
    nuclei_instance_patch_dir = save_dir / 'patches' / 'nuclei_instance'
    nuclei_class_patch_dir = save_dir / 'patches' / 'nuclei_class'
    tumor_patch_dir = save_dir / 'patches' / 'tumor'

    for wsi_path in wsi_paths:
        wsi_name = wsi_path.stem
        wsi_image_patch_dir = image_patch_dir / wsi_name
        wsi_nuclei_instance_patch_dir = nuclei_instance_patch_dir / wsi_name
        wsi_nuclei_class_patch_dir = nuclei_class_patch_dir / wsi_name
        wsi_tumor_patch_dir = tumor_patch_dir / wsi_name

        image_feats_save_path = save_dir / 'wsi_image_level' / f"{wsi_name}.csv"
        nuclei_feats_save_path = save_dir / 'wsi_nuclei_level' / f"{wsi_name}.csv"

        if image_feats_save_path.exists() and nuclei_feats_save_path.exists():
            continue

        image_patches = sorted(list(wsi_image_patch_dir.rglob("*.*")))
        nuclei_instance_patches = sorted(list(wsi_nuclei_instance_patch_dir.rglob("*.*")))
        nuclei_class_patches = sorted(list(wsi_nuclei_class_patch_dir.rglob("*.*")))
        tumor_patches = sorted(list(wsi_tumor_patch_dir.rglob("*.*")))

        # extract image and nuclei level feats from patches
        pbar = tqdm(total=len(image_patches), desc=f'Extracting {wsi_name}')
        def update_bar(*a):
            pbar.update()
        pool = Pool(n_workers)
        for image_patch, nuclei_instance_patch, nuclei_class_patch, tumor_patch in zip(image_patches, nuclei_instance_patches, nuclei_class_patches, tumor_patches):
            # print(image_patch.stem, nuclei_instance_patch.stem, nuclei_class_patch.stem, tumor_patch.stem)
            pool.apply_async(
                extract_patch_feats,
                args=(image_patch, nuclei_instance_patch, nuclei_class_patch, tumor_patch, save_dir, feature_type,
                      nuclei_type),
                callback=update_bar
            )
        # for image_patch, nuclei_instance_patch, nuclei_class_patch, tumor_patch in tqdm(
        #     zip(image_patches, nuclei_instance_patches, nuclei_class_patches, tumor_patches),
        #     desc=f'Extracting {wsi_name}'):
        #     print(image_patch.stem)
        #     extract_patch_feats(image_patch, nuclei_instance_patch, nuclei_class_patch, tumor_patch, save_dir)
    
        pool.close()
        pool.join()
        pbar.close()

        # merge
        image_feats_folder = save_dir / 'image_level' / wsi_name
        nuclei_feats_folder = save_dir / 'nuclei_level' / wsi_name
        
        merge_wsi_feats(image_feats_folder, image_feats_save_path)
        merge_wsi_feats(nuclei_feats_folder, nuclei_feats_save_path)

    return


def tissue_feats_extraction(wsi_dir, roi_dir, save_dir, mask_suffix="png", wsi_num=200, mag=40,
                            n_workers=25):
    """
    Feature extraction at the tissue (ROI) level, only "texture".

    :param wsi_dir: str, the directory of the folder where WSIs are stored
    :param roi_dir: str, the directory of roi mask (e.g., TLS, tumor, stroma, epithelial), grayscale image
    :param save_dir: str, str, the directory of saving results
    :param mask_suffix: str, the suffix of roi mask
    :param wsi_num: integer, [0, +inf), the selected number of WSIs for feature extraction
    :param mag: integer, target magnification for feature extraction
    :param n_workers: integer, the number of processes

    :return: csv file, feature matrix, each column represent a feature name, each row represents a patch
    """
    wsi_dir = Path(wsi_dir)
    roi_dir = Path(roi_dir)
    save_dir = Path(save_dir)

    wsi_paths = list(wsi_dir.glob("*.*"))
    # extracted_list_path = '/mnt/hd0/project/bca21gene/generated/fahzu/extracted_names.csv'
    # extracted_names = pd.read_csv(extracted_list_path)['name'].values

    # wsi_paths = [wsi_path for wsi_path in wsi_paths if wsi_path.stem not in extracted_names]
    # select index%2==0 for this pc
    # wsi_paths = wsi_paths[::2]
    if wsi_num:
        wsi_paths = wsi_paths[:wsi_num]
    else:
        wsi_paths = wsi_paths[:]

    # tqdm showing wsi name and progress, patch extraction
    for wsi_path in tqdm(wsi_paths, desc='Extracting Patches'):
        wsi_name = wsi_path.stem
        roi_path = roi_dir / f"{wsi_name}.{mask_suffix}"
        tissue_extract_patches(wsi_path, roi_path, save_dir, mag=mag, n_workers=n_workers)

    image_patch_dir = save_dir / 'patches' / 'image'
    roi_patch_dir = save_dir / 'patches' / 'roi'

    for wsi_path in wsi_paths:
        wsi_name = wsi_path.stem
        wsi_image_patch_dir = image_patch_dir / wsi_name
        wsi_roi_patch_dir = roi_patch_dir / wsi_name

        image_feats_save_path = save_dir / 'wsi_image_level' / f"{wsi_name}.csv"

        if image_feats_save_path.exists():
            continue

        image_patches = sorted(list(wsi_image_patch_dir.rglob("*.*")))
        roi_patches = sorted(list(wsi_roi_patch_dir.rglob("*.*")))

        # extract image and nuclei level feats from patches
        pbar = tqdm(total=len(image_patches), desc=f'Extracting {wsi_name}')

        def update_bar(*a):
            pbar.update()

        pool = Pool(n_workers)
        for image_patch, roi_patch in zip(image_patches,
                                          roi_patches):
            # print(image_patch.stem, nuclei_instance_patch.stem, nuclei_class_patch.stem, tumor_patch.stem)
            pool.apply_async(
                tissue_extract_patch_feats,
                args=(image_patch, roi_patch, save_dir),
                callback=update_bar
            )
        # for image_patch, nuclei_instance_patch, nuclei_class_patch, tumor_patch in tqdm(
        #     zip(image_patches, nuclei_instance_patches, nuclei_class_patches, tumor_patches),
        #     desc=f'Extracting {wsi_name}'):
        #     print(image_patch.stem)
        #     extract_patch_feats(image_patch, nuclei_instance_patch, nuclei_class_patch, tumor_patch, save_dir)

        pool.close()
        pool.join()
        pbar.close()

        # merge

        image_feats_folder = save_dir / 'image_level' / wsi_name

        merge_wsi_feats(image_feats_folder, image_feats_save_path)

    return


if __name__ == '__main__':
    ################## Nuclei feats extraction ##################
    # wsi_dir = Path("../example_folder/svs")
    # nuclei_dir = Path("../example_folder/nuclei_seg")
    # tumor_dir = Path("../example_folder/tumor_seg")
    # save_dir = Path("../example_folder/nuclei_feats_extraction")
    #
    # nuclei_feats_extraction(wsi_dir, nuclei_dir, tumor_dir, save_dir, mask_suffix="png", wsi_num=200,
    #                         mag=40, n_workers=3, feature_type="shape", nuclei_type=None)


    ################## Tissue feats extraction ##################
    wsi_dir = Path("../../example_folder/svs")
    roi_dir = Path("../../example_folder/roi_seg")
    save_dir = Path("../../example_folder/tissue_feats_extraction")
    tissue_feats_extraction(wsi_dir, roi_dir, save_dir, mask_suffix="png", wsi_num=200, mag=40,
                            n_workers=3)

