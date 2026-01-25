import numpy as np
import itertools
from shapely.geometry import Polygon
from pathomics.interplay.sptil_extractor.feat_statis_metrics import getFeatureStats
from pathomics.interplay.sptil_extractor.feat_statis_metrics import nchoosek

def getClusterDensityMeasures(groups):
    features = []
    featureNames = []
    numGroups = len(groups)
    stats = ['Total_', 'Mean_', 'Std_', 'Median_', 'Max_', 'Min_', 'Kurtosis_', 'Skewness_']
    meas = ['AreaClusters_G', 'DensityClusters_G']

    for i in range(numGroups):
        areas = groups[i]['clusterAreas']
        dens = groups[i]['Densities']

        if areas.size:
            features.append(getFeatureStats(areas))
            features.append(getFeatureStats(dens))
        else:
            features.append(np.zeros(16).tolist())

        featNam = ""
        for ms in range(len(meas)):
            for st in range(len(stats)):
                featNam = featNam + stats[st] + meas[ms] + str(i)
                featureNames.append(featNam)
                featNam = ""
    #### a little problem: the type of features is list with four list elements instead of a list with 16 float data
    features = np.array(features).reshape(1, -1)[0].tolist()
    return features, featureNames


def getClusterIntersectionMeasures(groups, maxNeibInter):
    features = []
    featureNames = []

    numGroups = len(groups)
    N = np.array(range(0, numGroups)).tolist()
    comb = nchoosek(N, 2)
    numComb = len(comb)

    for i in range(numComb):
        group0 = groups[comb[i][0]]
        group1 = groups[comb[i][1]]

        intAreas = []
        intRel0 = []
        intRel1 = []
        intRel2 = []

        g0 = str(comb[i][0])
        g1 = str(comb[i][1])
        stats = ['Total', 'Mean', 'Std', 'Median', 'Max', 'Min', 'Kurtosis', 'Skewness']
        meas = ['Intersected_AreaClusters_G' + g0 + '&' + g1,
                'RatioIntersected_AreaClusters_G' + g0 + '&' + g1 + '_ToArea_G' + g0,
                'RatioIntersected_AreaClusters_G' + g0 + '&' + g1 + '_ToArea_G' + g1,
                'RatioIntersected_AreaClusters_G' + g0 + '&' + g1 + '_ToAvgArea_G' + g0 + '&' + g1]

        for ms in meas:
            for st in stats:
                featureNames.append(st + ms)


        numClust0 = len(group0['clusters'])
        for j in range(numClust0):
            pol0 = group0['clusterPolygons'][j]
            area0 = group0['clusterAreas'][j]
            closestIdx = group0['neighborsPerGroup'][comb[i][1]]['closestIdx'][j]

            numClosest = len(closestIdx)
            if numClosest < maxNeibInter:
                maxNeibInter = numClosest

            numClust1 = len(group1['clusters'])
            if numClust1 < maxNeibInter:
                maxNeibInter = numClust1

            for k in range(maxNeibInter):
                pol1 = group1['clusterPolygons'][closestIdx[k]]
                area1 = group1['clusterAreas'][closestIdx[k]]
                plgn1 = Polygon(pol1)
                plgn0 = Polygon(pol0)
                intPoly = plgn0.intersection(plgn1)
                intArea = intPoly.area

                if intArea:
                    intAreas.append(intArea)
                    intRel0.append(intArea/area0)
                    intRel1.append(intArea/area1)
                    intRel2.append(2*intArea/(area0 + area1))

        if not len(intAreas):
            features.append(np.zeros(32))
        else:
            features.append(getFeatureStats(intAreas))
            features.append(getFeatureStats(intRel0))
            features.append(getFeatureStats(intRel1))
            features.append(getFeatureStats(intRel2))
    features = np.array(features).reshape(1, -1)[0].tolist()

    return features, featureNames


def getNeighborhoodMeasures(groups, neiborhoodsize, maxNumClusters): # todo nan
    features = []
    featureNames = []
    stats = ['Total', 'Mean', 'Std', 'Median', 'Max', 'Min', 'Kurtosis', 'Skewness']
    numStats = len(stats)
    # n = min(neiborhoodsize, maxNumClusters)
    n = neiborhoodsize
    numGroups = len(groups)

    for i in range(numGroups):
        numClust = len(groups[i]['clusters'])
        if not numClust:
            featStats = np.zeros(numGroups*neiborhoodsize*numStats).tolist()
        else:
            M = np.zeros((numClust, numGroups*neiborhoodsize))
            for j in range(numClust):
                colM = -1
                numRows = np.size(groups[i]['absoluteClosest'][j]['idx'], 0)
                if numRows:
                    for k in range(neiborhoodsize):
                        if k <= maxNumClusters and k <=numRows:
                            groupCompose = np.array(groups[i]['absoluteClosest'][j]['idx'][0:k+1, 0])
                            for coupleI in range(numGroups):
                                colM += 1
                                M[j, colM] = np.sum(groupCompose == coupleI)/(k+1)
                        else:
                            for coupleI in range(numGroups):
                                colM += 1
                                M[j, colM] = M[j, colM - numGroups]  ####does colM need +1 ?

            if numClust == 1:               #### lack of example to debug!!!
                stArr = getFeatureStats(M)  #### M or M.T ?
                featStats = np.zeros(numGroups*neiborhoodsize*numStats).tolist()
                numVal = len(stArr)
                for xx in range(numVal):
                    val = stArr[xx]
                    for yy in range(numStats):
                        ndx = xx + numVal*yy   ### or yy-1 ?
                        featStats[ndx] = val[yy]
            else:
                featStats = getFeatureStats(M)

        features.append(featStats)
        featStats = []

        for st in stats:
            for k in range(n):
                for coupleI in range(numGroups):
                    featureNames.append(st + 'PercentageClusters_G' + str(coupleI) + '_Surrounding_G' + str(i) + '_Neibor' + str(k))
    features = np.array(features).reshape(1, -1)[0].tolist()

    return features, featureNames