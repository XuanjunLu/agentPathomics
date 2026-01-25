#!/usr/bin/env python

from __future__ import print_function

import logging

import pathomics
from pathomics import featureextractor
from pathomics.image import *

# Get some test data

imageName, maskName = ('data/test2.png', 'histoqc')
image = Region(imageName, {'scale': 4})
coords = image.genPatchCoords()
print(coords)
