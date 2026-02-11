#!/usr/bin/env python

from __future__ import print_function
import os
import logging

import pathomics
from pathomics import featureextractor
import pandas as pd
from PIL import Image
import numpy as np

import multiprocessing
from multiprocessing import Pool, Manager
from itertools import repeat
from pathomics.helper import save_mat_mask, save_matfiles_info_to_df, select_image_mask_from_src, save_results_to_pandas
from pathomics.helper.preprocessing import get_low_magnification_WSI

# Get the Pypathomics logger (default log-level = INFO)
logger = pathomics.logger
logger.setLevel(
    logging.DEBUG
)  # set level to DEBUG to include debug log messages in log file

# Set up the handler to write out all log entries to a file
handler = logging.FileHandler(filename='testLog.txt', mode='w')
formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Define settings for signature calculation
# These are currently set equal to the respective default values
settings = {}


def run(imageName, maskName, features):
    # Initialize feature extractor
    extractor = featureextractor.PathomicsFeatureExtractor(**settings)

    # Disable all classes except histoqc
    extractor.disableAllFeatures()

    # Only enable morph in nuclei
    extractor.enableFeaturesByName(**features)

    featureVector = extractor.execute(imageName, maskName)
    return featureVector


if __name__ == '__main__':

    # Step 1: set enabled features and basic parameters
    features = dict(firstorder=[], glcm=[], glrlm=[])
    n_workers = 4
    base_dir = '../pathomics_prepare/1319263-8'
    image_path = f'{base_dir}/1319263-8.svs'
    patient_id = '1319263-8'
    mask_path = f'{base_dir}/1319263-8_mask.png'
    save_dir = f'{base_dir}/example2'
    os.makedirs(save_dir, exist_ok=True)

    # Step 2: get low magnification image file
    low_mag_wsi_save_path = f'{save_dir}/{patient_id}.jpg'
    max_magnification = 40
    save_magnification = 2.5
    get_low_magnification_WSI(image_path, max_magnification,
                              save_magnification, low_mag_wsi_save_path)
    image_path = low_mag_wsi_save_path

    # Step 4: compute features by multiprocessing
    # can load as numpy array
    image = Image.open(image_path)
    image = np.array(image)
    mask = Image.open(mask_path)
    mask = np.array(mask)
    image_path = image
    mask_path = mask
    featureVector = run(image_path, mask_path, features)

    # Step 5: save results
    save_results_csv_path = f'{save_dir}/{patient_id}_2.5x_tumorbed_feat.csv'
    data = {}
    data['patient_id'] = patient_id
    for feature_name, feature_value in featureVector.items():
        if data.get(feature_name) is None:
            data[feature_name] = [feature_value]
        else:
            data[feature_name].append(feature_value)

    df = save_results_to_pandas(data, save_results_csv_path)
