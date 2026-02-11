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
from pathomics.helper import extract_tumorNuclear_coords_from_one_mat
# from .matlab import *
from pathomics.interplay.sptil_extractor import extract_SpaTIL_feature


class PathomicsSpaTIL(base.PathomicsFeaturesBase):

    def __init__(self, inputImage, inputMask, **kwargs):
        # check mask length
        assert len(inputMask) == 2, "SpaTIL need two masks, one for tumor and one for lymph"
        super(PathomicsSpaTIL, self).__init__(inputImage, inputMask[0], **kwargs)
        self.name = 'SpaTIL'
        self.image = sitk.GetArrayViewFromImage(self.inputImage)
        
        self.mask = [] 

        for m in inputMask:
            _mask = sitk.GetArrayViewFromImage(m)
        
            if len(_mask.shape) > 2:
                _mask = _mask[:, :, 0]
            self.mask.append(_mask)

        tumor_bounds, _, _ = self.mask2bounds(
            self.mask[0], self.image, atts=['area'], **kwargs)
        lymph_bounds, _, _ = self.mask2bounds(
            self.mask[1], self.image, atts=['area'], **kwargs)

        tumor_coords = self.bounds2coords(tumor_bounds)
        lymph_coords = self.bounds2coords(lymph_bounds)

        self.features = extract_SpaTIL_feature(tumor_coords, lymph_coords)

    def getNumClustersG1FeatureValue(self):
        return self.features[0]

    def getNumClustersG2FeatureValue(self):
        return self.features[1]

    def getTotalAreaClustersG1FeatureValue(self):
        return self.features[2]

    def getMeanAreaClustersG1FeatureValue(self):
        return self.features[3]

    def getStdAreaClustersG1FeatureValue(self):
        return self.features[4]

    def getMedianAreaClustersG1FeatureValue(self):
        return self.features[5]

    def getMaxAreaClustersG1FeatureValue(self):
        return self.features[6]

    def getMinAreaClustersG1FeatureValue(self):
        return self.features[7]

    def getKurtosisAreaClustersG1FeatureValue(self):
        return self.features[8]

    def getSkewnessAreaClustersG1FeatureValue(self):
        return self.features[9]

    def getTotalDensityClustersG1FeatureValue(self):
        return self.features[10]

    def getMeanDensityClustersG1FeatureValue(self):
        return self.features[11]

    def getStdDensityClustersG1FeatureValue(self):
        return self.features[12]

    def getMedianDensityClustersG1FeatureValue(self):
        return self.features[13]

    def getMaxDensityClustersG1FeatureValue(self):
        return self.features[14]

    def getMinDensityClustersG1FeatureValue(self):
        return self.features[15]

    def getKurtosisDensityClustersG1FeatureValue(self):
        return self.features[16]

    def getSkewnessDensityClustersG1FeatureValue(self):
        return self.features[17]

    def getTotalAreaClustersG2FeatureValue(self):
        return self.features[18]

    def getMeanAreaClustersG2FeatureValue(self):
        return self.features[19]

    def getStdAreaClustersG2FeatureValue(self):
        return self.features[20]

    def getMedianAreaClustersG2FeatureValue(self):
        return self.features[21]

    def getMaxAreaClustersG2FeatureValue(self):
        return self.features[22]

    def getMinAreaClustersG2FeatureValue(self):
        return self.features[23]

    def getKurtosisAreaClustersG2FeatureValue(self):
        return self.features[24]

    def getSkewnessAreaClustersG2FeatureValue(self):
        return self.features[25]

    def getTotalDensityClustersG2FeatureValue(self):
        return self.features[26]

    def getMeanDensityClustersG2FeatureValue(self):
        return self.features[27]

    def getStdDensityClustersG2FeatureValue(self):
        return self.features[28]

    def getMedianDensityClustersG2FeatureValue(self):
        return self.features[29]

    def getMaxDensityClustersG2FeatureValue(self):
        return self.features[30]

    def getMinDensityClustersG2FeatureValue(self):
        return self.features[31]

    def getKurtosisDensityClustersG2FeatureValue(self):
        return self.features[32]

    def getSkewnessDensityClustersG2FeatureValue(self):
        return self.features[33]

    def getTotalIntersectedAreaClustersG1and2FeatureValue(self):
        return self.features[34]

    def getMeanIntersectedAreaClustersG1and2FeatureValue(self):
        return self.features[35]

    def getStdIntersectedAreaClustersG1and2FeatureValue(self):
        return self.features[36]

    def getMedianIntersectedAreaClustersG1and2FeatureValue(self):
        return self.features[37]

    def getMaxIntersectedAreaClustersG1and2FeatureValue(self):
        return self.features[38]

    def getMinIntersectedAreaClustersG1and2FeatureValue(self):
        return self.features[39]

    def getKurtosisIntersectedAreaClustersG1and2FeatureValue(self):
        return self.features[40]

    def getSkewnessIntersectedAreaClustersG1and2FeatureValue(self):
        return self.features[41]

    def getTotalRatioIntersectedAreaClustersG1and2ToAreaG1FeatureValue(self):
        return self.features[42]

    def getMeanRatioIntersectedAreaClustersG1and2ToAreaG1FeatureValue(self):
        return self.features[43]

    def getStdRatioIntersectedAreaClustersG1and2ToAreaG1FeatureValue(self):
        return self.features[44]

    def getMedianRatioIntersectedAreaClustersG1and2ToAreaG1FeatureValue(self):
        return self.features[45]

    def getMaxRatioIntersectedAreaClustersG1and2ToAreaG1FeatureValue(self):
        return self.features[46]

    def getMinRatioIntersectedAreaClustersG1and2ToAreaG1FeatureValue(self):
        return self.features[47]

    def getKurtosisRatioIntersectedAreaClustersG1and2ToAreaG1FeatureValue(self):
        return self.features[48]

    def getSkewnessRatioIntersectedAreaClustersG1and2ToAreaG1FeatureValue(self):
        return self.features[49]

    def getTotalRatioIntersectedAreaClustersG1and2ToAreaG2FeatureValue(self):
        return self.features[50]

    def getMeanRatioIntersectedAreaClustersG1and2ToAreaG2FeatureValue(self):
        return self.features[51]

    def getStdRatioIntersectedAreaClustersG1and2ToAreaG2FeatureValue(self):
        return self.features[52]

    def getMedianRatioIntersectedAreaClustersG1and2ToAreaG2FeatureValue(self):
        return self.features[53]

    def getMaxRatioIntersectedAreaClustersG1and2ToAreaG2FeatureValue(self):
        return self.features[54]

    def getMinRatioIntersectedAreaClustersG1and2ToAreaG2FeatureValue(self):
        return self.features[55]

    def getKurtosisRatioIntersectedAreaClustersG1and2ToAreaG2FeatureValue(self):
        return self.features[56]

    def getSkewnessRatioIntersectedAreaClustersG1and2ToAreaG2FeatureValue(self):
        return self.features[57]

    def getTotalRatioIntersectedAreaClustersG1and2ToAvgAreaG1and2FeatureValue(self):
        return self.features[58]

    def getMeanRatioIntersectedAreaClustersG1and2ToAvgAreaG1and2FeatureValue(self):
        return self.features[59]

    def getStdRatioIntersectedAreaClustersG1and2ToAvgAreaG1and2FeatureValue(self):
        return self.features[60]

    def getMedianRatioIntersectedAreaClustersG1and2ToAvgAreaG1and2FeatureValue(self):
        return self.features[61]

    def getMaxRatioIntersectedAreaClustersG1and2ToAvgAreaG1and2FeatureValue(self):
        return self.features[62]

    def getMinRatioIntersectedAreaClustersG1and2ToAvgAreaG1and2FeatureValue(self):
        return self.features[63]

    def getKurtosisRatioIntersectedAreaClustersG1and2ToAvgAreaG1and2FeatureValue(self):
        return self.features[64]

    def getSkewnessRatioIntersectedAreaClustersG1and2ToAvgAreaG1and2FeatureValue(self):
        return self.features[65]

    def getTotalPercentageClustersG1SurroundingG1Neighborhood1FeatureValue(self):
        return self.features[66]

    def getTotalPercentageClustersG2SurroundingG1Neighborhood1FeatureValue(self):
        return self.features[67]

    def getTotalPercentageClustersG1SurroundingG1Neighborhood2FeatureValue(self):
        return self.features[68]

    def getTotalPercentageClustersG2SurroundingG1Neighborhood2FeatureValue(self):
        return self.features[69]

    def getTotalPercentageClustersG1SurroundingG1Neighborhood3FeatureValue(self):
        return self.features[70]

    def getTotalPercentageClustersG2SurroundingG1Neighborhood3FeatureValue(self):
        return self.features[71]

    def getTotalPercentageClustersG1SurroundingG1Neighborhood4FeatureValue(self):
        return self.features[72]

    def getTotalPercentageClustersG2SurroundingG1Neighborhood4FeatureValue(self):
        return self.features[73]

    def getTotalPercentageClustersG1SurroundingG1Neighborhood5FeatureValue(self):
        return self.features[74]

    def getTotalPercentageClustersG2SurroundingG1Neighborhood5FeatureValue(self):
        return self.features[75]

    def getMeanPercentageClustersG1SurroundingG1Neighborhood1FeatureValue(self):
        return self.features[76]

    def getMeanPercentageClustersG2SurroundingG1Neighborhood1FeatureValue(self):
        return self.features[77]

    def getMeanPercentageClustersG1SurroundingG1Neighborhood2FeatureValue(self):
        return self.features[78]

    def getMeanPercentageClustersG2SurroundingG1Neighborhood2FeatureValue(self):
        return self.features[79]

    def getMeanPercentageClustersG1SurroundingG1Neighborhood3FeatureValue(self):
        return self.features[80]

    def getMeanPercentageClustersG2SurroundingG1Neighborhood3FeatureValue(self):
        return self.features[81]

    def getMeanPercentageClustersG1SurroundingG1Neighborhood4FeatureValue(self):
        return self.features[82]

    def getMeanPercentageClustersG2SurroundingG1Neighborhood4FeatureValue(self):
        return self.features[83]

    def getMeanPercentageClustersG1SurroundingG1Neighborhood5FeatureValue(self):
        return self.features[84]

    def getMeanPercentageClustersG2SurroundingG1Neighborhood5FeatureValue(self):
        return self.features[85]

    def getStdPercentageClustersG1SurroundingG1Neighborhood1FeatureValue(self):
        return self.features[86]

    def getStdPercentageClustersG2SurroundingG1Neighborhood1FeatureValue(self):
        return self.features[87]

    def getStdPercentageClustersG1SurroundingG1Neighborhood2FeatureValue(self):
        return self.features[88]

    def getStdPercentageClustersG2SurroundingG1Neighborhood2FeatureValue(self):
        return self.features[89]

    def getStdPercentageClustersG1SurroundingG1Neighborhood3FeatureValue(self):
        return self.features[90]

    def getStdPercentageClustersG2SurroundingG1Neighborhood3FeatureValue(self):
        return self.features[91]

    def getStdPercentageClustersG1SurroundingG1Neighborhood4FeatureValue(self):
        return self.features[92]

    def getStdPercentageClustersG2SurroundingG1Neighborhood4FeatureValue(self):
        return self.features[93]

    def getStdPercentageClustersG1SurroundingG1Neighborhood5FeatureValue(self):
        return self.features[94]

    def getStdPercentageClustersG2SurroundingG1Neighborhood5FeatureValue(self):
        return self.features[95]

    def getMedianPercentageClustersG1SurroundingG1Neighborhood1FeatureValue(self):
        return self.features[96]

    def getMedianPercentageClustersG2SurroundingG1Neighborhood1FeatureValue(self):
        return self.features[97]

    def getMedianPercentageClustersG1SurroundingG1Neighborhood2FeatureValue(self):
        return self.features[98]

    def getMedianPercentageClustersG2SurroundingG1Neighborhood2FeatureValue(self):
        return self.features[99]

    def getMedianPercentageClustersG1SurroundingG1Neighborhood3FeatureValue(self):
        return self.features[100]

    def getMedianPercentageClustersG2SurroundingG1Neighborhood3FeatureValue(self):
        return self.features[101]

    def getMedianPercentageClustersG1SurroundingG1Neighborhood4FeatureValue(self):
        return self.features[102]

    def getMedianPercentageClustersG2SurroundingG1Neighborhood4FeatureValue(self):
        return self.features[103]

    def getMedianPercentageClustersG1SurroundingG1Neighborhood5FeatureValue(self):
        return self.features[104]

    def getMedianPercentageClustersG2SurroundingG1Neighborhood5FeatureValue(self):
        return self.features[105]

    def getMaxPercentageClustersG1SurroundingG1Neighborhood1FeatureValue(self):
        return self.features[106]

    def getMaxPercentageClustersG2SurroundingG1Neighborhood1FeatureValue(self):
        return self.features[107]

    def getMaxPercentageClustersG1SurroundingG1Neighborhood2FeatureValue(self):
        return self.features[108]

    def getMaxPercentageClustersG2SurroundingG1Neighborhood2FeatureValue(self):
        return self.features[109]

    def getMaxPercentageClustersG1SurroundingG1Neighborhood3FeatureValue(self):
        return self.features[110]

    def getMaxPercentageClustersG2SurroundingG1Neighborhood3FeatureValue(self):
        return self.features[111]

    def getMaxPercentageClustersG1SurroundingG1Neighborhood4FeatureValue(self):
        return self.features[112]

    def getMaxPercentageClustersG2SurroundingG1Neighborhood4FeatureValue(self):
        return self.features[113]

    def getMaxPercentageClustersG1SurroundingG1Neighborhood5FeatureValue(self):
        return self.features[114]

    def getMaxPercentageClustersG2SurroundingG1Neighborhood5FeatureValue(self):
        return self.features[115]

    def getMinPercentageClustersG1SurroundingG1Neighborhood1FeatureValue(self):
        return self.features[116]

    def getMinPercentageClustersG2SurroundingG1Neighborhood1FeatureValue(self):
        return self.features[117]

    def getMinPercentageClustersG1SurroundingG1Neighborhood2FeatureValue(self):
        return self.features[118]

    def getMinPercentageClustersG2SurroundingG1Neighborhood2FeatureValue(self):
        return self.features[119]

    def getMinPercentageClustersG1SurroundingG1Neighborhood3FeatureValue(self):
        return self.features[120]

    def getMinPercentageClustersG2SurroundingG1Neighborhood3FeatureValue(self):
        return self.features[121]

    def getMinPercentageClustersG1SurroundingG1Neighborhood4FeatureValue(self):
        return self.features[122]

    def getMinPercentageClustersG2SurroundingG1Neighborhood4FeatureValue(self):
        return self.features[123]

    def getMinPercentageClustersG1SurroundingG1Neighborhood5FeatureValue(self):
        return self.features[124]

    def getMinPercentageClustersG2SurroundingG1Neighborhood5FeatureValue(self):
        return self.features[125]

    def getKurtosisPercentageClustersG1SurroundingG1Neighborhood1FeatureValue(self):
        return self.features[126]

    def getKurtosisPercentageClustersG2SurroundingG1Neighborhood1FeatureValue(self):
        return self.features[127]

    def getKurtosisPercentageClustersG1SurroundingG1Neighborhood2FeatureValue(self):
        return self.features[128]

    def getKurtosisPercentageClustersG2SurroundingG1Neighborhood2FeatureValue(self):
        return self.features[129]

    def getKurtosisPercentageClustersG1SurroundingG1Neighborhood3FeatureValue(self):
        return self.features[130]

    def getKurtosisPercentageClustersG2SurroundingG1Neighborhood3FeatureValue(self):
        return self.features[131]

    def getKurtosisPercentageClustersG1SurroundingG1Neighborhood4FeatureValue(self):
        return self.features[132]

    def getKurtosisPercentageClustersG2SurroundingG1Neighborhood4FeatureValue(self):
        return self.features[133]

    def getKurtosisPercentageClustersG1SurroundingG1Neighborhood5FeatureValue(self):
        return self.features[134]

    def getKurtosisPercentageClustersG2SurroundingG1Neighborhood5FeatureValue(self):
        return self.features[135]

    def getSkewnessPercentageClustersG1SurroundingG1Neighborhood1FeatureValue(self):
        return self.features[136]

    def getSkewnessPercentageClustersG2SurroundingG1Neighborhood1FeatureValue(self):
        return self.features[137]

    def getSkewnessPercentageClustersG1SurroundingG1Neighborhood2FeatureValue(self):
        return self.features[138]

    def getSkewnessPercentageClustersG2SurroundingG1Neighborhood2FeatureValue(self):
        return self.features[139]

    def getSkewnessPercentageClustersG1SurroundingG1Neighborhood3FeatureValue(self):
        return self.features[140]

    def getSkewnessPercentageClustersG2SurroundingG1Neighborhood3FeatureValue(self):
        return self.features[141]

    def getSkewnessPercentageClustersG1SurroundingG1Neighborhood4FeatureValue(self):
        return self.features[142]

    def getSkewnessPercentageClustersG2SurroundingG1Neighborhood4FeatureValue(self):
        return self.features[143]

    def getSkewnessPercentageClustersG1SurroundingG1Neighborhood5FeatureValue(self):
        return self.features[144]

    def getSkewnessPercentageClustersG2SurroundingG1Neighborhood5FeatureValue(self):
        return self.features[145]

    def getTotalPercentageClustersG1SurroundingG2Neighborhood1FeatureValue(self):
        return self.features[146]

    def getTotalPercentageClustersG2SurroundingG2Neighborhood1FeatureValue(self):
        return self.features[147]

    def getTotalPercentageClustersG1SurroundingG2Neighborhood2FeatureValue(self):
        return self.features[148]

    def getTotalPercentageClustersG2SurroundingG2Neighborhood2FeatureValue(self):
        return self.features[149]

    def getTotalPercentageClustersG1SurroundingG2Neighborhood3FeatureValue(self):
        return self.features[150]

    def getTotalPercentageClustersG2SurroundingG2Neighborhood3FeatureValue(self):
        return self.features[151]

    def getTotalPercentageClustersG1SurroundingG2Neighborhood4FeatureValue(self):
        return self.features[152]

    def getTotalPercentageClustersG2SurroundingG2Neighborhood4FeatureValue(self):
        return self.features[153]

    def getTotalPercentageClustersG1SurroundingG2Neighborhood5FeatureValue(self):
        return self.features[154]

    def getTotalPercentageClustersG2SurroundingG2Neighborhood5FeatureValue(self):
        return self.features[155]

    def getMeanPercentageClustersG1SurroundingG2Neighborhood1FeatureValue(self):
        return self.features[156]

    def getMeanPercentageClustersG2SurroundingG2Neighborhood1FeatureValue(self):
        return self.features[157]

    def getMeanPercentageClustersG1SurroundingG2Neighborhood2FeatureValue(self):
        return self.features[158]

    def getMeanPercentageClustersG2SurroundingG2Neighborhood2FeatureValue(self):
        return self.features[159]

    def getMeanPercentageClustersG1SurroundingG2Neighborhood3FeatureValue(self):
        return self.features[160]

    def getMeanPercentageClustersG2SurroundingG2Neighborhood3FeatureValue(self):
        return self.features[161]

    def getMeanPercentageClustersG1SurroundingG2Neighborhood4FeatureValue(self):
        return self.features[162]

    def getMeanPercentageClustersG2SurroundingG2Neighborhood4FeatureValue(self):
        return self.features[163]

    def getMeanPercentageClustersG1SurroundingG2Neighborhood5FeatureValue(self):
        return self.features[164]

    def getMeanPercentageClustersG2SurroundingG2Neighborhood5FeatureValue(self):
        return self.features[165]

    def getStdPercentageClustersG1SurroundingG2Neighborhood1FeatureValue(self):
        return self.features[166]

    def getStdPercentageClustersG2SurroundingG2Neighborhood1FeatureValue(self):
        return self.features[167]

    def getStdPercentageClustersG1SurroundingG2Neighborhood2FeatureValue(self):
        return self.features[168]

    def getStdPercentageClustersG2SurroundingG2Neighborhood2FeatureValue(self):
        return self.features[169]

    def getStdPercentageClustersG1SurroundingG2Neighborhood3FeatureValue(self):
        return self.features[170]

    def getStdPercentageClustersG2SurroundingG2Neighborhood3FeatureValue(self):
        return self.features[171]

    def getStdPercentageClustersG1SurroundingG2Neighborhood4FeatureValue(self):
        return self.features[172]

    def getStdPercentageClustersG2SurroundingG2Neighborhood4FeatureValue(self):
        return self.features[173]

    def getStdPercentageClustersG1SurroundingG2Neighborhood5FeatureValue(self):
        return self.features[174]

    def getStdPercentageClustersG2SurroundingG2Neighborhood5FeatureValue(self):
        return self.features[175]

    def getMedianPercentageClustersG1SurroundingG2Neighborhood1FeatureValue(self):
        return self.features[176]

    def getMedianPercentageClustersG2SurroundingG2Neighborhood1FeatureValue(self):
        return self.features[177]

    def getMedianPercentageClustersG1SurroundingG2Neighborhood2FeatureValue(self):
        return self.features[178]

    def getMedianPercentageClustersG2SurroundingG2Neighborhood2FeatureValue(self):
        return self.features[179]

    def getMedianPercentageClustersG1SurroundingG2Neighborhood3FeatureValue(self):
        return self.features[180]

    def getMedianPercentageClustersG2SurroundingG2Neighborhood3FeatureValue(self):
        return self.features[181]

    def getMedianPercentageClustersG1SurroundingG2Neighborhood4FeatureValue(self):
        return self.features[182]

    def getMedianPercentageClustersG2SurroundingG2Neighborhood4FeatureValue(self):
        return self.features[183]

    def getMedianPercentageClustersG1SurroundingG2Neighborhood5FeatureValue(self):
        return self.features[184]

    def getMedianPercentageClustersG2SurroundingG2Neighborhood5FeatureValue(self):
        return self.features[185]

    def getMaxPercentageClustersG1SurroundingG2Neighborhood1FeatureValue(self):
        return self.features[186]

    def getMaxPercentageClustersG2SurroundingG2Neighborhood1FeatureValue(self):
        return self.features[187]

    def getMaxPercentageClustersG1SurroundingG2Neighborhood2FeatureValue(self):
        return self.features[188]

    def getMaxPercentageClustersG2SurroundingG2Neighborhood2FeatureValue(self):
        return self.features[189]

    def getMaxPercentageClustersG1SurroundingG2Neighborhood3FeatureValue(self):
        return self.features[190]

    def getMaxPercentageClustersG2SurroundingG2Neighborhood3FeatureValue(self):
        return self.features[191]

    def getMaxPercentageClustersG1SurroundingG2Neighborhood4FeatureValue(self):
        return self.features[192]

    def getMaxPercentageClustersG2SurroundingG2Neighborhood4FeatureValue(self):
        return self.features[193]

    def getMaxPercentageClustersG1SurroundingG2Neighborhood5FeatureValue(self):
        return self.features[194]

    def getMaxPercentageClustersG2SurroundingG2Neighborhood5FeatureValue(self):
        return self.features[195]

    def getMinPercentageClustersG1SurroundingG2Neighborhood1FeatureValue(self):
        return self.features[196]

    def getMinPercentageClustersG2SurroundingG2Neighborhood1FeatureValue(self):
        return self.features[197]

    def getMinPercentageClustersG1SurroundingG2Neighborhood2FeatureValue(self):
        return self.features[198]

    def getMinPercentageClustersG2SurroundingG2Neighborhood2FeatureValue(self):
        return self.features[199]

    def getMinPercentageClustersG1SurroundingG2Neighborhood3FeatureValue(self):
        return self.features[200]

    def getMinPercentageClustersG2SurroundingG2Neighborhood3FeatureValue(self):
        return self.features[201]

    def getMinPercentageClustersG1SurroundingG2Neighborhood4FeatureValue(self):
        return self.features[202]

    def getMinPercentageClustersG2SurroundingG2Neighborhood4FeatureValue(self):
        return self.features[203]

    def getMinPercentageClustersG1SurroundingG2Neighborhood5FeatureValue(self):
        return self.features[204]

    def getMinPercentageClustersG2SurroundingG2Neighborhood5FeatureValue(self):
        return self.features[205]

    def getKurtosisPercentageClustersG1SurroundingG2Neighborhood1FeatureValue(self):
        return self.features[206]

    def getKurtosisPercentageClustersG2SurroundingG2Neighborhood1FeatureValue(self):
        return self.features[207]

    def getKurtosisPercentageClustersG1SurroundingG2Neighborhood2FeatureValue(self):
        return self.features[208]

    def getKurtosisPercentageClustersG2SurroundingG2Neighborhood2FeatureValue(self):
        return self.features[209]

    def getKurtosisPercentageClustersG1SurroundingG2Neighborhood3FeatureValue(self):
        return self.features[210]

    def getKurtosisPercentageClustersG2SurroundingG2Neighborhood3FeatureValue(self):
        return self.features[211]

    def getKurtosisPercentageClustersG1SurroundingG2Neighborhood4FeatureValue(self):
        return self.features[212]

    def getKurtosisPercentageClustersG2SurroundingG2Neighborhood4FeatureValue(self):
        return self.features[213]

    def getKurtosisPercentageClustersG1SurroundingG2Neighborhood5FeatureValue(self):
        return self.features[214]

    def getKurtosisPercentageClustersG2SurroundingG2Neighborhood5FeatureValue(self):
        return self.features[215]

    def getSkewnessPercentageClustersG1SurroundingG2Neighborhood1FeatureValue(self):
        return self.features[216]

    def getSkewnessPercentageClustersG2SurroundingG2Neighborhood1FeatureValue(self):
        return self.features[217]

    def getSkewnessPercentageClustersG1SurroundingG2Neighborhood2FeatureValue(self):
        return self.features[218]

    def getSkewnessPercentageClustersG2SurroundingG2Neighborhood2FeatureValue(self):
        return self.features[219]

    def getSkewnessPercentageClustersG1SurroundingG2Neighborhood3FeatureValue(self):
        return self.features[220]

    def getSkewnessPercentageClustersG2SurroundingG2Neighborhood3FeatureValue(self):
        return self.features[221]

    def getSkewnessPercentageClustersG1SurroundingG2Neighborhood4FeatureValue(self):
        return self.features[222]

    def getSkewnessPercentageClustersG2SurroundingG2Neighborhood4FeatureValue(self):
        return self.features[223]

    def getSkewnessPercentageClustersG1SurroundingG2Neighborhood5FeatureValue(self):
        return self.features[224]

    def getSkewnessPercentageClustersG2SurroundingG2Neighborhood5FeatureValue(self):
        return self.features[225]

    def getTotalGroupingFactorG1FeatureValue(self):
        return self.features[226]

    def getMeanGroupingFactorG1FeatureValue(self):
        return self.features[227]

    def getStdGroupingFactorG1FeatureValue(self):
        return self.features[228]

    def getMedianGroupingFactorG1FeatureValue(self):
        return self.features[229]

    def getMaxGroupingFactorG1FeatureValue(self):
        return self.features[230]

    def getMinGroupingFactorG1FeatureValue(self):
        return self.features[231]

    def getKurtosisGroupingFactorG1FeatureValue(self):
        return self.features[232]

    def getSkewnessGroupingFactorG1FeatureValue(self):
        return self.features[233]

    def getTotalGroupingFactorG2FeatureValue(self):
        return self.features[234]

    def getMeanGroupingFactorG2FeatureValue(self):
        return self.features[235]

    def getStdGroupingFactorG2FeatureValue(self):
        return self.features[236]

    def getMedianGroupingFactorG2FeatureValue(self):
        return self.features[237]

    def getMaxGroupingFactorG2FeatureValue(self):
        return self.features[238]

    def getMinGroupingFactorG2FeatureValue(self):
        return self.features[239]

    def getKurtosisGroupingFactorG2FeatureValue(self):
        return self.features[240]

    def getSkewnessGroupingFactorG2FeatureValue(self):
        return self.features[241]

    def getIntersectionAreaG1and2FeatureValue(self):
        return self.features[242]

    def getRatioIntersectedAreaG1and2ToAreaG1FeatureValue(self):
        return self.features[243]

    def getRatioIntersectedAreaG1and2ToAreaG2FeatureValue(self):
        return self.features[244]

    def getRatioIntersectedAreaG1and2ToAvgAreaG1and2FeatureValue(self):
        return self.features[245]

    def getNumCentroidsClustersG1InConvHullG2FeatureValue(self):
        return self.features[246]

    def getNumCentroidsClustersG2InConvHullG1FeatureValue(self):
        return self.features[247]

    def getGraphAreaStandardDeviationG1FeatureValue(self):
        return self.features[248]

    def getGraphAreaAverageG1FeatureValue(self):
        return self.features[249]

    def getGraphAreaMinimumorMaximumG1FeatureValue(self):
        return self.features[250]

    def getGraphAreaDisorderG1FeatureValue(self):
        return self.features[251]

    def getGraphPerimeterStandardDeviationG1FeatureValue(self):
        return self.features[252]

    def getGraphPerimeterAverageG1FeatureValue(self):
        return self.features[253]

    def getGraphPerimeterMinimumorMaximumG1FeatureValue(self):
        return self.features[254]

    def getGraphPerimeterDisorderG1FeatureValue(self):
        return self.features[255]

    def getGraphChordStandardDeviationG1FeatureValue(self):
        return self.features[256]

    def getGraphChordAverageG1FeatureValue(self):
        return self.features[257]

    def getGraphChordMinimumorMaximumG1FeatureValue(self):
        return self.features[258]

    def getGraphChordDisorderG1FeatureValue(self):
        return self.features[259]

    def getGraphSideLengthMinimumorMaximumG1FeatureValue(self):
        return self.features[260]

    def getGraphSideLengthStandardDeviationG1FeatureValue(self):
        return self.features[261]

    def getGraphSideLengthAverageG1FeatureValue(self):
        return self.features[262]

    def getGraphSideLengthDisorderG1FeatureValue(self):
        return self.features[263]

    def getGraphTriangleAreaMinimumorMaximumG1FeatureValue(self):
        return self.features[264]

    def getGraphTriangleAreaStandardDeviationG1FeatureValue(self):
        return self.features[265]

    def getGraphTriangleAreaAverageG1FeatureValue(self):
        return self.features[266]

    def getGraphTriangleAreaDisorderG1FeatureValue(self):
        return self.features[267]

    def getGraphMSTEdgeLengthAverageG1FeatureValue(self):
        return self.features[268]

    def getGraphMSTEdgeLengthStandardDeviationG1FeatureValue(self):
        return self.features[269]

    def getGraphMSTEdgeLengthMinimumorMaximumG1FeatureValue(self):
        return self.features[270]

    def getGraphMSTEdgeLengthDisorderG1FeatureValue(self):
        return self.features[271]

    def getGraphAreaofpolygonsG1FeatureValue(self):
        return self.features[272]

    def getGraphNumberofnucleiG1FeatureValue(self):
        return self.features[273]

    def getGraphDensityofNucleiG1FeatureValue(self):
        return self.features[274]

    def getGraphAveragedistanceto3NearestNeighborsG1FeatureValue(self):
        return self.features[275]

    def getGraphAveragedistanceto5NearestNeighborsG1FeatureValue(self):
        return self.features[276]

    def getGraphAveragedistanceto7NearestNeighborsG1FeatureValue(self):
        return self.features[277]

    def getGraphStandardDeviationdistanceto3NearestNeighborsG1FeatureValue(self):
        return self.features[278]

    def getGraphStandardDeviationdistanceto5NearestNeighborsG1FeatureValue(self):
        return self.features[279]

    def getGraphStandardDeviationdistanceto7NearestNeighborsG1FeatureValue(self):
        return self.features[280]

    def getGraphDisorderofdistanceto3NearestNeighborsG1FeatureValue(self):
        return self.features[281]

    def getGraphDisorderofdistanceto5NearestNeighborsG1FeatureValue(self):
        return self.features[282]

    def getGraphDisorderofdistanceto7NearestNeighborsG1FeatureValue(self):
        return self.features[283]

    def getGraphAvgNearestNeighborsina10PixelRadiusG1FeatureValue(self):
        return self.features[284]

    def getGraphAvgNearestNeighborsina20PixelRadiusG1FeatureValue(self):
        return self.features[285]

    def getGraphAvgNearestNeighborsina30PixelRadiusG1FeatureValue(self):
        return self.features[286]

    def getGraphAvgNearestNeighborsina40PixelRadiusG1FeatureValue(self):
        return self.features[287]

    def getGraphAvgNearestNeighborsina50PixelRadiusG1FeatureValue(self):
        return self.features[288]

    def getGraphStandardDeviationNearestNeighborsina10PixelRadiusG1FeatureValue(self):
        return self.features[289]

    def getGraphStandardDeviationNearestNeighborsina20PixelRadiusG1FeatureValue(self):
        return self.features[290]

    def getGraphStandardDeviationNearestNeighborsina30PixelRadiusG1FeatureValue(self):
        return self.features[291]

    def getGraphStandardDeviationNearestNeighborsina40PixelRadiusG1FeatureValue(self):
        return self.features[292]

    def getGraphStandardDeviationNearestNeighborsina50PixelRadiusG1FeatureValue(self):
        return self.features[293]

    def getGraphDisorderofNearestNeighborsina10PixelRadiusG1FeatureValue(self):
        return self.features[294]

    def getGraphDisorderofNearestNeighborsina20PixelRadiusG1FeatureValue(self):
        return self.features[295]

    def getGraphDisorderofNearestNeighborsina30PixelRadiusG1FeatureValue(self):
        return self.features[296]

    def getGraphDisorderofNearestNeighborsina40PixelRadiusG1FeatureValue(self):
        return self.features[297]

    def getGraphDisorderofNearestNeighborsina50PixelRadiusG1FeatureValue(self):
        return self.features[298]

    def getGraphAreaStandardDeviationG2FeatureValue(self):
        return self.features[299]

    def getGraphAreaAverageG2FeatureValue(self):
        return self.features[300]

    def getGraphAreaMinimumorMaximumG2FeatureValue(self):
        return self.features[301]

    def getGraphAreaDisorderG2FeatureValue(self):
        return self.features[302]

    def getGraphPerimeterStandardDeviationG2FeatureValue(self):
        return self.features[303]

    def getGraphPerimeterAverageG2FeatureValue(self):
        return self.features[304]

    def getGraphPerimeterMinimumorMaximumG2FeatureValue(self):
        return self.features[305]

    def getGraphPerimeterDisorderG2FeatureValue(self):
        return self.features[306]

    def getGraphChordStandardDeviationG2FeatureValue(self):
        return self.features[307]

    def getGraphChordAverageG2FeatureValue(self):
        return self.features[308]

    def getGraphChordMinimumorMaximumG2FeatureValue(self):
        return self.features[309]

    def getGraphChordDisorderG2FeatureValue(self):
        return self.features[310]

    def getGraphSideLengthMinimumorMaximumG2FeatureValue(self):
        return self.features[311]

    def getGraphSideLengthStandardDeviationG2FeatureValue(self):
        return self.features[312]

    def getGraphSideLengthAverageG2FeatureValue(self):
        return self.features[313]

    def getGraphSideLengthDisorderG2FeatureValue(self):
        return self.features[314]

    def getGraphTriangleAreaMinimumorMaximumG2FeatureValue(self):
        return self.features[315]

    def getGraphTriangleAreaStandardDeviationG2FeatureValue(self):
        return self.features[316]

    def getGraphTriangleAreaAverageG2FeatureValue(self):
        return self.features[317]

    def getGraphTriangleAreaDisorderG2FeatureValue(self):
        return self.features[318]

    def getGraphMSTEdgeLengthAverageG2FeatureValue(self):
        return self.features[319]

    def getGraphMSTEdgeLengthStandardDeviationG2FeatureValue(self):
        return self.features[320]

    def getGraphMSTEdgeLengthMinimumorMaximumG2FeatureValue(self):
        return self.features[321]

    def getGraphMSTEdgeLengthDisorderG2FeatureValue(self):
        return self.features[322]

    def getGraphAreaofpolygonsG2FeatureValue(self):
        return self.features[323]

    def getGraphNumberofnucleiG2FeatureValue(self):
        return self.features[324]

    def getGraphDensityofNucleiG2FeatureValue(self):
        return self.features[325]

    def getGraphAveragedistanceto3NearestNeighborsG2FeatureValue(self):
        return self.features[326]

    def getGraphAveragedistanceto5NearestNeighborsG2FeatureValue(self):
        return self.features[327]

    def getGraphAveragedistanceto7NearestNeighborsG2FeatureValue(self):
        return self.features[328]

    def getGraphStandardDeviationdistanceto3NearestNeighborsG2FeatureValue(self):
        return self.features[329]

    def getGraphStandardDeviationdistanceto5NearestNeighborsG2FeatureValue(self):
        return self.features[330]

    def getGraphStandardDeviationdistanceto7NearestNeighborsG2FeatureValue(self):
        return self.features[331]

    def getGraphDisorderofdistanceto3NearestNeighborsG2FeatureValue(self):
        return self.features[332]

    def getGraphDisorderofdistanceto5NearestNeighborsG2FeatureValue(self):
        return self.features[333]

    def getGraphDisorderofdistanceto7NearestNeighborsG2FeatureValue(self):
        return self.features[334]

    def getGraphAvgNearestNeighborsina10PixelRadiusG2FeatureValue(self):
        return self.features[335]

    def getGraphAvgNearestNeighborsina20PixelRadiusG2FeatureValue(self):
        return self.features[336]

    def getGraphAvgNearestNeighborsina30PixelRadiusG2FeatureValue(self):
        return self.features[337]

    def getGraphAvgNearestNeighborsina40PixelRadiusG2FeatureValue(self):
        return self.features[338]

    def getGraphAvgNearestNeighborsina50PixelRadiusG2FeatureValue(self):
        return self.features[339]

    def getGraphStandardDeviationNearestNeighborsina10PixelRadiusG2FeatureValue(self):
        return self.features[340]

    def getGraphStandardDeviationNearestNeighborsina20PixelRadiusG2FeatureValue(self):
        return self.features[341]

    def getGraphStandardDeviationNearestNeighborsina30PixelRadiusG2FeatureValue(self):
        return self.features[342]

    def getGraphStandardDeviationNearestNeighborsina40PixelRadiusG2FeatureValue(self):
        return self.features[343]

    def getGraphStandardDeviationNearestNeighborsina50PixelRadiusG2FeatureValue(self):
        return self.features[344]

    def getGraphDisorderofNearestNeighborsina10PixelRadiusG2FeatureValue(self):
        return self.features[345]

    def getGraphDisorderofNearestNeighborsina20PixelRadiusG2FeatureValue(self):
        return self.features[346]

    def getGraphDisorderofNearestNeighborsina30PixelRadiusG2FeatureValue(self):
        return self.features[347]

    def getGraphDisorderofNearestNeighborsina40PixelRadiusG2FeatureValue(self):
        return self.features[348]

    def getGraphDisorderofNearestNeighborsina50PixelRadiusG2FeatureValue(self):
        return self.features[349]
