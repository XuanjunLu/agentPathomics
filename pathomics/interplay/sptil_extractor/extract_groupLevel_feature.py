import numpy as np
from scipy.spatial.distance import pdist, squareform
from pathomics.interplay.sptil_extractor.feat_statis_metrics import getFeatureStats
from pathomics.interplay.sptil_extractor.feat_statis_metrics import nchoosek
from matplotlib.path import Path
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from pathomics.topological.globalgraph import get_graph_features


def getSumNodeWeightsThreshold(feature, distMethod, threshold):
    dist = pdist(feature, distMethod)
    # dist = squareform(dist)
    dist[dist == 0.0] = 1.0
    dist = 1/dist
    dist[dist < threshold] = 0
    dist = squareform(dist)

    vector = np.zeros(len(dist))
    for i in range(len(dist)):
        vector[i] = sum(dist[i])
    return vector


def getGroupingFactorByGroup(groups, groupingThreshold):
    features = []
    featureNames = []
    stats = ['Total', 'Mean', 'Std', 'Median', 'Max', 'Min', 'Kurtosis', 'Skewness']
    numStats = len(stats)
    numGroups = len(groups)

    feature = []
    for i in range(numGroups):
        feaVector = getSumNodeWeightsThreshold(groups[i]['clusterCentroids'], 'euclidean', groupingThreshold)
        feature = getFeatureStats(feaVector)
        features.append(feature)
        feature = []

        for st in stats:
            featureNames.append(st + '_GroupingFactor_G' + str(i))
    features = np.array(features).reshape(1, -1)[0].tolist()

    return features, featureNames


def getIntersectionGroups(groups):
    features = []
    featureNames = []

    numGroups = len(groups)
    N = np.array(range(0, numGroups)).tolist()
    comb = nchoosek(N, 2)
    numComb = len(comb)

    G_polygons = dict()
    for i in range(numGroups):
        G_polygons[i] = dict()
        points = groups[i]['clusterCentroids']
        if len(points) > 2:
            hull = ConvexHull(points)
            G_polygons[i]['coords'] = points[hull.vertices]
            G_polygons[i]['area'] = hull.volume
        else:
            G_polygons[i]['coords'] = []
            G_polygons[i]['area'] = 0

    for i in range(numComb):
        g0 = comb[i][0]
        g1 = comb[i][1]

        pol0 = G_polygons[g0]['coords']
        pol1 = G_polygons[g1]['coords']
        if len(pol0) and len(pol1):
            intPoly = Polygon(pol0).intersection(Polygon(pol1))
            intArea = intPoly.area
            avgArea = (G_polygons[g0]['area'] + G_polygons[g1]['area'])/2

            g0_in_p1 = Path(pol1).contains_points(groups[g0]['clusterCentroids'], radius=-1e-10)  # in1
            g1_in_p0 = Path(pol0).contains_points(groups[g1]['clusterCentroids'], radius=-1e-10)   # in2

            feature = [intArea, intArea/G_polygons[g0]['area'], intArea/G_polygons[g1]['area'], intArea/avgArea , np.sum(g0_in_p1 == 1), np.sum(g1_in_p0 == 1)]
            features.append(feature)
            feature = []
        else:
            features.append([0, 0, 0, 0, 0, 0])

        featureNames = ['IntersectionArea_G' + str(g0) + '&' + str(g1),
                        'RatioIntersectedArea_G' + str(g0) + '&' + str(g1) + '_ToArea_G' + str(g0),
                        'RatioIntersectedArea_G' + str(g0) + '&' + str(g1) + '_ToArea_G' + str(g1),
                        'RatioIntersectedArea_G' + str(g0) + '&' + str(g1) + '_ToAvgArea_G' + str(g0) + '&' + str(g1),
                        'NumCentroidsClusters_G' + str(0) + '_InConvHull_G' + str(g1),
                        'NumCentroidsClusters_G' + str(1) + '_InConvHull_G' + str(g0)]    #### or featureNames.append(featureName) ?
    features = np.array(features).reshape(1, -1)[0].tolist()

    return features, featureNames


def getCltGraphFeature(groups):
    features = []
    featureNames = []
    numGroups = len(groups)

    featureName = ['Area_Standard_Deviation', 'Area_Average', 'Area_Minimum/Maximum',
                   'Area_Disorder', 'Perimeter_Standard_Deviation', 'Perimeter_Average',
                   'Perimeter_Minimum/Maximum', 'Perimeter_Disorder',  'Chord_Standard_Deviation',
                   'Chord_Average', 'Chord_Minimum/Maximum', 'Chord_Disorder',
                   'Side_Length_Minimum/Maximum', 'Side_Length_Standard_Deviation', 'Side_Length_Average',
                   'Side_Length_Disorder', 'Triangle_Area_Minimum/Maximum', 'Triangle_Area_Standard_Deviation',
                   'Triangle_Area_Average', 'Triangle_Area_Disorder', 'MST_Edge_Length_Average',
                   'MST_Edge_Length_Standard_Deviation', 'MST_Edge_Length_Minimum/Maximum', 'MST_Edge_Length_Disorder',
                   'Area_of_polygons', 'Number_of_nuclei', 'Density_of_Nuclei',
                   'Average_distance_to_3_Nearest_Neighbors', 'Average_distance_to_5_Nearest_Neighbors', 'Average_distance_to_7_Nearest_Neighbors',
                   'Standard_Deviation_distance_to_3_Nearest_Neighbors', 'Standard_Deviation_distance_to_5_Nearest_Neighbors', 'Standard_Deviation_distance_to_7_Nearest_Neighbors',
                   'Disorder_of_distance_to_3_Nearest_Neighbors', 'Disorder_of_distance_to_5_Nearest Neighbors', 'Disorder_of_distance_to_7_Nearest Neighbors',
                   'Avg_Nearest_Neighbors_in_a_10_Pixel_Radius', 'Avg_Nearest_Neighbors_in_a_20_Pixel_Radius', 'Avg_Nearest_Neighbors_in_a_30_Pixel_Radius',
                   'Avg_Nearest_Neighbors_in_a_40_Pixel_Radius', 'Avg_Nearest_Neighbors_in_a_50_Pixel_Radius', 'Standard_Deviation_Nearest_Neighbors_in_a_10_Pixel Radius',
                   'Standard_Deviation_Nearest_Neighbors_in_a_20_Pixel_Radius', 'Standard_Deviation_Nearest_Neighbors_in_a_30_Pixel_Radius', 'Standard_Deviation_Nearest_Neighbors_in_a_40_Pixel_Radius',
                   'Standard_Deviation_Nearest_Neighbors_in_a_50_Pixel_Radius', 'Disorder_of_Nearest_Neighbors_in_a_10_Pixel_Radius', 'Disorder_of_Nearest_Neighbors_in_a_20_Pixel_Radius',
                   'Disorder_of_Nearest_Neighbors_in_a_30_Pixel_Radius', 'Disorder_of_Nearest_Neighbors_in_a_40_Pixel_Radius', 'Disorder_of_Nearest_Neighbors_in_a_50_Pixel_Radius']

    for i in range(numGroups):
        centroids = groups[i]['clusterCentroids'].tolist()
        if np.size(centroids, 0) > 2:
            feature = get_graph_features(centroids)
            features.append(feature)
        else:
            feature = np.zeros(len(featureName))
            features.append(feature)

        for name in featureName:
            featureNames.append('cltGraph_' + name + '_G' + str(i))
    features = np.array(features).reshape(1, -1)[0].tolist()

    return features, featureNames




