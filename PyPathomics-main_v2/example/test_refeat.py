#!/usr/bin/env python

from __future__ import print_function

import logging

import pathomics
from pathomics import featureextractor
import pandas as pd

import multiprocessing
from multiprocessing import Pool, Manager
from itertools import repeat
import numpy as np

# Get the Pypathomics logger (default log-level = INFO)
# logger = pathomics.logger
# logger.setLevel(
#     logging.DEBUG
# )  # set level to DEBUG to include debug log messages in log file

# # Set up the handler to write out all log entries to a file
# handler = logging.FileHandler(filename='testLog.txt', mode='w')
# formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
# handler.setFormatter(formatter)
# logger.addHandler(handler)

# Define settings for signature calculation
# These are currently set equal to the respective default values
# settings = {}


def run(imageName, maskName):
    # Initialize feature extractor
    extractor = featureextractor.PathomicsFeatureExtractor()

    # Disable features, default all enabled 
    extractor.disableAllFeatures()

    # Only enable morph in nuclei
    extractor.enableFeaturesByName(
        # firstorder=[
        # ],
        # glcm=[
        #     # 'Autocorrelation',
        #     # 'JointAverage',
        #     # 'Imc1',
        #     # 'Imc2',
        # ],
        # glrlm=[
        #     # 'ShortRunEmphasis',
        # ],
        # glszm = [

        # ],
        # ngtdm = [
        # ],
        # gldm = [],
        # contourbased = [
        #     # 'AreaRatio',
        #     # 'DistanceRatio',
        #     # 'StdDistance',
        #     # 'VarianceDistance',
        #     # 'InvariantMoment1',
        #     # 'FractalDimension',
        # ],
        # basic = [
        #     # 'Area',
        # ]
        # rgt=[],
        # localgraph = [
        #     # 'AverageEccentricity90percent',
        # ],
        # clustergraph=[],
        # flock=[],
        # globalgraph = [],
        # flock=[],
        spatil=[],
    )
    featureVector = extractor.execute(imageName, maskName)
    return featureVector


if __name__ == '__main__':
    img_file = './data/image1.png'
    mask_file = './data/image1_mask.png'
    # img_file = './data/1319263-8_19044_44361_10x.png'
    # mask_file = './data/1319263-8_19044_44361_10x-mask.png'
    mask_file = [mask_file]*2
    featureVectors = run(img_file, mask_file)
    for feature_name, feature_value in featureVectors.items():
        try:
            # check if numpy array
            if isinstance(feature_value, np.ndarray) and feature_value.ndim > 0:
                # print(feature_name, feature_value[:6], feature_value.shape)
                print(feature_name.split('_')[-1])
            else:
                # print(feature_name, feature_value)
                print(feature_name.split('_')[-1])
            # print(feature_name, feature_value)
        except Exception as e:
            print(feature_name, 'Error')
            # print error message
            print(e)
            print(feature_name, feature_value, type(feature_value), isinstance(feature_value, np.ndarray))

