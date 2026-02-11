import numpy as np
import pandas as pd
# import sys
# sys.path.append('./sptil_extractor')
# from pathomics.matlab.sptil_extractor import *

from pathomics.interplay.sptil_extractor.build_cluster import *
from pathomics.interplay.sptil_extractor.identify_cluster_neighbor import *
from pathomics.interplay.sptil_extractor.extract_clusterLevel_feature import *
from pathomics.interplay.sptil_extractor.extract_groupLevel_feature import *


def extract_SpaTIL_feature(tumor_coords, lymph_coords):  #### node0, node1
    alpha = 0.42
    r = 0.185
    neigborhoodSize = 5
    groupingThreshold = 0.0005

    features = []
    featureNames = []

    groups = dict()
    groups[0] = dict()
    groups[1] = dict()
    numGroups = len(groups)


    groups[0]['nodes'] = tumor_coords
    groups[1]['nodes'] = lymph_coords

    clustPerGroup = np.zeros(numGroups)
    for i in range(numGroups):
        edges = getCCGEdges(groups[i]['nodes'], alpha, r)
        groups[i]['numClust'], sizes, groups[i]['clusters'] = networkComponments(edges)
        groups[i]['clusterCentroids'], groups[i]['clusterPolygons'], groups[i]['clusterAreas'], groups[i][
            'Densities'] = getClusterProperties(groups[i]['clusters'], groups[i]['nodes'])
        clustPerGroup[i] = groups[i]['numClust']

        features.append(clustPerGroup[i])
        featureNames.append('NumClusters_G' + str(i))

    for i in range(numGroups):
        groups[i]['neighborsPerGroup'] = getClosestNeigborByGroup(groups, i)

    maxNumClust = int(np.max(clustPerGroup))
    for i in range(numGroups):
        groups[i]['absoluteClosest'] = getAbsoluteClosestNeighbors(groups, i, maxNumClust)

    densFeat, densFeatNames = getClusterDensityMeasures(groups)
    intersClustFeat, intersClustFeatNames = getClusterIntersectionMeasures(groups, neigborhoodSize)
    richFeat, richFeatNames = getNeighborhoodMeasures(groups, neigborhoodSize, maxNumClust)

    groupingFeat, groupingFeatNames = getGroupingFactorByGroup(groups, groupingThreshold)
    intersGroupFeat, intersGroupFeatNames = getIntersectionGroups(groups)
    graphFeat, graphFeatNames = getCltGraphFeature(groups)

    features.append(densFeat)
    features.append(intersClustFeat)
    features.append(richFeat)
    features.append(groupingFeat)
    features.append(intersGroupFeat)
    features.append(graphFeat)

    featureValues = []
    for j in features:
        if type(j) == list:
            for k in j:
                featureValues.append(k)
        else:
            featureValues.append(j)

    featureNames.append(densFeatNames)
    featureNames.append(intersClustFeatNames)
    featureNames.append(richFeatNames)##
    featureNames.append(groupingFeatNames)
    featureNames.append(intersGroupFeatNames)
    featureNames.append(graphFeatNames)

    featNam = []
    for j in featureNames:
        if type(j) == list:
            for k in j:
                featNam.append(k)
        else:
            featNam.append(j)

    feat_df = pd.DataFrame([featureValues])
    feat_df.columns = featNam
    feat_df.index = ['feature_data']

    # name_and_value = dict()
    # for i in range(len(featNam)):
    #     name_and_value[featNam[i]] = featureValues[i]

    return featureValues


