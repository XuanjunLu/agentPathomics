import numpy
from six.moves import range
import scipy.io as sio
import SimpleITK as sitk
from PIL import Image
import numpy as np
import skimage
from skimage import measure
from skimage import morphology
import pandas as pd
from pathomics import base
from PIL import Image
import numpy as np
from skimage import measure


class PathomicsBasic(base.PathomicsFeaturesBase):

    def __init__(self, inputImage, inputMask, **kwargs):
        super(PathomicsBasic, self).__init__(inputImage, inputMask,
                                                **kwargs)
        self.name = 'Basic'
        self.image = sitk.GetArrayViewFromImage(self.inputImage)
        self.mask = sitk.GetArrayViewFromImage(self.inputMask)
        if len(self.mask.shape) > 2:
            self.mask = self.mask[:, :, 0]
        self.bounds, self.image_intensity, self.feats = self.mask2bounds(
            self.mask, self.image, atts=['area'], **kwargs)
        
        # self.features, _ = extract_nuclei_props(
        #                     fname_mask=self.mask, fname_intensity=self.image_intensity)
        
        
        labels = measure.label(self.mask)

        # image_intensity = Image.open(fname_intensity).convert('L')
        image_intensity = Image.fromarray(self.image_intensity).convert('L')
        image_intensity = np.array(image_intensity)
        self.props = measure.regionprops(label_image=labels, intensity_image=image_intensity)
        
        # atts = ['area', 'axis_major_length', 'axis_minor_length', 'eccentricity', 'equivalent_diameter_area', 
        #         'euler_number', 'extent', 'feret_diameter_max', 'orientation', 'perimeter', 'perimeter_crofton', 
        #         'solidity', 'intensity_max', 'intensity_mean', 'intensity_min']
        # feats_name=['centroid_x', 'centroid_y'] + atts + ['intensity_std']

        # feats = []
        # for prop in props: 
        #     feats.append([prop.centroid[1], prop.centroid[0]] + [prop[att] for att in atts] + [np.std(prop['image_intensity'])])
        
    def getCentroidXFeatureValue(self):
        return np.array([prop.centroid[1] for prop in self.props])

    def getCentroidYFeatureValue(self):
        return np.array([prop.centroid[0] for prop in self.props])

    def getAreaFeatureValue(self):
        return np.array([prop.area for prop in self.props])

    def getAxisMajorLengthFeatureValue(self):
        return np.array([prop.axis_major_length for prop in self.props])

    def getAxisMinorLengthFeatureValue(self):
        return np.array([prop.axis_minor_length for prop in self.props])

    def getEccentricityFeatureValue(self):
        return np.array([prop.eccentricity for prop in self.props])

    def getEquivalentDiameterAreaFeatureValue(self):
        return np.array([prop.equivalent_diameter_area for prop in self.props])

    def getEulerNumberFeatureValue(self):
        return np.array([prop.euler_number for prop in self.props])

    def getExtentFeatureValue(self):
        return np.array([prop.extent for prop in self.props])

    def getFeretDiameterMaxFeatureValue(self):
        return np.array([prop.feret_diameter_max for prop in self.props])

    def getOrientationFeatureValue(self):
        return np.array([prop.orientation for prop in self.props])

    def getPerimeterFeatureValue(self):
        return np.array([prop.perimeter for prop in self.props])

    def getPerimeterCroftonFeatureValue(self):
        return np.array([prop.perimeter_crofton for prop in self.props])

    def getSolidityFeatureValue(self):
        return np.array([prop.solidity for prop in self.props])

    def getIntensityMaxFeatureValue(self):
        return np.array([prop.intensity_max for prop in self.props])

    def getIntensityMeanFeatureValue(self):
        return np.array([prop.intensity_mean for prop in self.props])

    def getIntensityMinFeatureValue(self):
        return np.array([prop.intensity_min for prop in self.props])

    def getIntensityStdFeatureValue(self):
        return np.array([np.std(prop.intensity_image) for prop in self.props])
