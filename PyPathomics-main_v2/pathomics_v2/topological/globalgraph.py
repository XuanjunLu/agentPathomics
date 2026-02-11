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

from re import X
from turtle import distance
from uuid import RESERVED_FUTURE
import numpy as np
import random
import scipy
from scipy import spatial
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

def list_flatten(t):
    ##make a flat list from a list of lists
    return [item for sublist in t for item in sublist]

def eucl(p1, p2):
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    dis_matrix = scipy.spatial.distance.cdist(p1, p2, 'euclidean')
    return dis_matrix


def polyarea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def featureCalBasis(a_list):
    a1 = np.std(np.array(a_list), ddof=1)
    a2 = np.mean(np.array(a_list))
    a3 = np.min(np.array(a_list)) / np.max(np.array(a_list))
    a4 = 1 - (1 / (1 + a1 / a2))  ##what is it meaning for??
    return [a1, a2, a3, a4]


def get_graph_features(x_y_coord):
    """
    x = [bounds[i][0] for i in range(len(bounds))]
    y = [bounds[i][1] for i in range(len(bounds))]
    x_coord = [bounds[i][4] for i in range(len(bounds))]
    y_coord = [bounds[i][5] for i in range(len(bounds))]

    # % Voronoi Diagram Features
    # % Area  ???
    x_y_coord = list(map(list, zip(x_coord, y_coord)))
    """
    coords = np.array(x_y_coord)
    x_coord = coords[:, 0]
    y_coord = coords[:, 1]

    vor = scipy.spatial.Voronoi(x_y_coord)
    V = vor.vertices
    C = vor.regions  #### sorted. but MATLAB version is in random order.
    C = C[1:]

    # # Find point coordinate for each region
    # sorting = [np.where(vor.point_region==x)[0][0] for x in range(1, len(vor.regions))]
    # # sort regions along coordinate list `sorting` (exclude first, empty list [])
    # sorted_regions = [x for _, x in sorted(zip(sorting, vor.regions[1:]))]

    # C_=[]
    # for Cc in C:
    #     Cc = [cc + 2 for cc in Cc]
    #     Cc = np.clip(Cc, 0, V.shape[0]-1)
    #     Cc = Cc.astype('int')
    #     Cc = Cc.tolist()
    #     C_.append(Cc)
    # C = C_

    # # Find point coordinate for each region
    # sorting = [np.where(vor.point_region==vor_x)[0][0] for vor_x in range(1, len(vor.regions))]
    # # sort regions along coordinate list `sorting` (exclude first, empty list [])
    # C = [vor_x for _, vor_x in sorted(zip(sorting, vor.regions[1:]))]

    # X_test = [[1915.48997468393, 2997.17338884367],
    # [1949.99602663184, 3033.57771401122],
    # [1949.84391701610, 3056.84272097757],
    # [1914.53417789789, 3057.59312542984],
    # [1907.07163069539, 3002.66638196300]]

    # # print(dir(vor))
    chorddist = []
    perimdist = []
    area = []

    for i in range(len(C)):
        if(len(C[i]) > 2):   ####### why occur empty list in C ????
            X = V[C[i], :]
            # X = np.array(X_test)
            chorddist_mat = np.zeros(shape=[len(X), len(X)])
            for ii in range(len(X)):
                chorddist_mat[ii, :] = np.linalg.norm(X - X[ii], axis=1)
            chorddist_ = [
                chorddist_mat[j, k] for j in range(len(X))
                for k in range(j + 1, len(X))
            ]
            perimdist_ = [chorddist_mat[j, j + 1] for j in range(len(X) - 1)
                          ] + [chorddist_mat[0, len(X) - 1]]
            chorddist.append(chorddist_)
            perimdist.append(perimdist_)
            area.append(polyarea(X[:, 0], X[:, 1]))

    chorddist, perimdist = list_flatten(chorddist), list_flatten(perimdist)

    vfeature = featureCalBasis(area) + featureCalBasis(
        perimdist) + featureCalBasis(chorddist)

    import pandas as pd
    df_vfeature = pd.DataFrame()
    for i in range(len(vfeature)):
        df_vfeature['vfeature{}'.format(i)] = vfeature[i]

    # # Delaunay
    # # Edge length and area
    tri = scipy.spatial.Delaunay(x_y_coord)
    # print(tri)
    delau = tri.simplices.copy()

    delau = delau.tolist()
    sidelen = []
    triarea = []
    for i in range(len(delau)):
        sidelen_ = []

        t = [[x_coord[ind], y_coord[ind]] for ind in delau[i]]
        t = np.array(t)
        sidelen_.append(np.linalg.norm(t[0] - t[1]))
        sidelen_.append(np.linalg.norm(t[0] - t[2]))
        sidelen_.append(np.linalg.norm(t[1] - t[2]))
        sidelen.append(sidelen_)
        triarea.append(polyarea(t[:, 0], t[:, 1]))

    # # vfeature(13) - vfeature(16)  no problems
    sidelen = list_flatten(sidelen)
    vfeat13 = np.min(sidelen) / np.max(sidelen)
    vfeat14 = np.std(np.array(sidelen), ddof=1)
    vfeat15 = np.mean(np.array(sidelen))
    vfeat16 = 1 - (1 / (1 + vfeat14 / vfeat15))  ##what is it meaning for??
    vfeature13_16 = [vfeat13, vfeat14, vfeat15, vfeat16]

    vfeat17 = np.min(triarea) / np.max(triarea)
    vfeat18 = np.std(np.array(triarea), ddof=1)
    vfeat19 = np.mean(np.array(triarea))
    vfeat20 = 1 - (1 / (1 + vfeat18 / vfeat19))  ##what is it meaning for??

    vfeature17_20 = [vfeat17, vfeat18, vfeat19, vfeat20]

    # % MST: Average MST Edge Length
    # % The MST is a tree that spans the entire population in such a way that the
    # % sum of the Euclidian edge length is minimal.

    ## cal the distance matrix
    distmat = eucl(x_y_coord, x_y_coord)
    # highval = np.amax(distmat)+1
    # print(np.amax(distmat))

    ## keep Upper triangle  Undirected graph
    distmat_triu = np.triu(distmat)
    ## minimum spanning tree
    Tcsr = scipy.sparse.csgraph.minimum_spanning_tree(distmat_triu)
    # print(dir(Tcsr))
    mst_edges = Tcsr.indices
    mst_edgelen = Tcsr.data
    mst_totlen = np.sum(mst_edgelen)
    # Tcsr = Tcsr.toarray()
    ###no problems
    vfeat21 = np.mean(np.array(mst_edgelen))
    vfeat22 = np.std(np.array(mst_edgelen), ddof=1)
    vfeat23 = np.min(mst_edgelen) / np.max(mst_edgelen)
    vfeat24 = 1 - (1 / (1 + vfeat22 / vfeat21))  ##what is it meaning for??

    # vfeature21_24 = vfeature17_20 + featureCalBasis(mst_edgelen)
    vfeature21_24 = [vfeat21, vfeat22, vfeat23, vfeat24]
    # % Nuclear Features
    # % Density
    ### have problem???
    # vfeature25_27 = vfeature21_24 + [sum(area), len(C), len(C)/sum(area) ]
    vfeat25 = sum(area)
    vfeat26 = len(C)
    vfeat27 = vfeat26 / vfeat25
    vfeature25_27 = [vfeat25, vfeat26, vfeat27]

    # % Average Distance to K-NN
    # % Construct N x N distance matrix:
    K = [3, 5, 7]
    distmat_s = np.sort(distmat)
    DKNN = [np.sum(distmat_s[:, 1:K_i + 1], axis=-1) for K_i in K]

    vfeat28 = np.mean(np.array(DKNN[0]))
    vfeat29 = np.mean(np.array(DKNN[1]))
    vfeat30 = np.mean(np.array(DKNN[2]))
    vfeat31 = np.std(np.array(DKNN[0]), ddof=1)
    vfeat32 = np.std(np.array(DKNN[1]), ddof=1)
    vfeat33 = np.std(np.array(DKNN[2]), ddof=1)
    vfeat34 = 1 - (1 / (1 + vfeat31 / vfeat28))  ##what is it meaning for??
    vfeat35 = 1 - (1 / (1 + vfeat32 / vfeat29))  ##what is it meaning for??
    vfeat36 = 1 - (1 / (1 + vfeat33 / vfeat30))  ##what is it meaning for??

    vfeature28_36 = [
        vfeat28, vfeat29, vfeat30, vfeat31, vfeat32, vfeat33, vfeat34, vfeat35,
        vfeat36
    ]

    # vf_l = []
    # for i in range(len(K)):
    #     vf = featureCalBasis(DKNN[i])
    #     del vf[2]
    #     vf_l = vf_l + vf
    # vfeature28_36 = vfeature25_27 + vf_l

    # % NNRR_av: Average Number of Neighbors in a Restricted Radius
    # % Set the number of pixels within which to search
    NNRR = []
    R = [10, 20, 30, 40, 50]
    for restr in R:
        NNRR.append([
            sum((distmat[i, :] < restr) * 1) - 1 for i in range(len(distmat))
        ])

    vfeat37 = np.mean(np.array(NNRR[0])) if sum(NNRR[0]) != 0 else 0
    vfeat38 = np.mean(np.array(NNRR[1])) if sum(NNRR[1]) != 0 else 0
    vfeat39 = np.mean(np.array(NNRR[2])) if sum(NNRR[2]) != 0 else 0
    vfeat40 = np.mean(np.array(NNRR[3])) if sum(NNRR[3]) != 0 else 0
    vfeat41 = np.mean(np.array(NNRR[4])) if sum(NNRR[4]) != 0 else 0

    vfeat42 = np.std(np.array(NNRR[0]), ddof=1) if sum(NNRR[0]) != 0 else 0
    vfeat43 = np.std(np.array(NNRR[1]), ddof=1) if sum(NNRR[1]) != 0 else 0
    vfeat44 = np.std(np.array(NNRR[2]), ddof=1) if sum(NNRR[2]) != 0 else 0
    vfeat45 = np.std(np.array(NNRR[3]), ddof=1) if sum(NNRR[3]) != 0 else 0
    vfeat46 = np.std(np.array(NNRR[4]), ddof=1) if sum(NNRR[4]) != 0 else 0

    vfeat47 = 1 - (1 / (1 + vfeat42 / vfeat37)) if sum(NNRR[0]) != 0 else 0
    vfeat48 = 1 - (1 / (1 + vfeat43 / vfeat38)) if sum(NNRR[1]) != 0 else 0
    vfeat49 = 1 - (1 / (1 + vfeat44 / vfeat39)) if sum(NNRR[2]) != 0 else 0
    vfeat50 = 1 - (1 / (1 + vfeat45 / vfeat40)) if sum(NNRR[3]) != 0 else 0
    vfeat51 = 1 - (1 / (1 + vfeat46 / vfeat41)) if sum(NNRR[4]) != 0 else 0

    vfeature37_51 = [
        vfeat37,
        vfeat38,
        vfeat39,
        vfeat40,
        vfeat41,
        vfeat42,
        vfeat43,
        vfeat44,
        vfeat45,
        vfeat46,
        vfeat47,
        vfeat48,
        vfeat49,
        vfeat50,
        vfeat51,
    ]

    vfeature = vfeature + vfeature13_16 + vfeature17_20 + vfeature21_24 + vfeature25_27 + vfeature28_36 + vfeature37_51
    return vfeature



class PathomicsGlobalGraph(base.PathomicsFeaturesBase):

    def __init__(self, inputImage, inputMask, **kwargs):
        super(PathomicsGlobalGraph, self).__init__(inputImage, inputMask, **kwargs)
        self.name = 'GlobalGraph'
        self.image = sitk.GetArrayViewFromImage(self.inputImage)
        self.mask = sitk.GetArrayViewFromImage(self.inputMask)
        if len(self.mask.shape) > 2:
            self.mask = self.mask[:, :, 0]
        self.bounds, self.image_intensity, self.feats = self.mask2bounds(
            self.mask, self.image, atts=['area'], **kwargs)

        coords = self.bounds2coords(self.bounds)
        self.features = get_graph_features(coords)

    def getAreaStandardDeviationFeatureValue(self):
        return self.features[0]

    def getAreaAverageFeatureValue(self):
        return self.features[1]

    def getAreaMinimumOrMaximumFeatureValue(self):
        return self.features[2]

    def getAreaDisorderFeatureValue(self):
        return self.features[3]

    def getPerimeterStandardDeviationFeatureValue(self):
        return self.features[4]

    def getPerimeterAverageFeatureValue(self):
        return self.features[5]

    def getPerimeterMinimumOrMaximumFeatureValue(self):
        return self.features[6]

    def getPerimeterDisorderFeatureValue(self):
        return self.features[7]

    def getChordStandardDeviationFeatureValue(self):
        return self.features[8]

    def getChordAverageFeatureValue(self):
        return self.features[9]

    def getChordMinimumOrMaximumFeatureValue(self):
        return self.features[10]

    def getChordDisorderFeatureValue(self):
        return self.features[11]

    def getSideLengthMinimumOrMaximumFeatureValue(self):
        return self.features[12]

    def getSideLengthStandardDeviationFeatureValue(self):
        return self.features[13]

    def getSideLengthAverageFeatureValue(self):
        return self.features[14]

    def getSideLengthDisorderFeatureValue(self):
        return self.features[15]

    def getTriangleAreaMinimumOrMaximumFeatureValue(self):
        return self.features[16]

    def getTriangleAreaStandardDeviationFeatureValue(self):
        return self.features[17]

    def getTriangleAreaAverageFeatureValue(self):
        return self.features[18]

    def getTriangleAreaDisorderFeatureValue(self):
        return self.features[19]

    def getMSTEdgeLengthAverageFeatureValue(self):
        return self.features[20]

    def getMSTEdgeLengthStandardDeviationFeatureValue(self):
        return self.features[21]

    def getMSTEdgeLengthMinimumOrMaximumFeatureValue(self):
        return self.features[22]

    def getMSTEdgeLengthDisorderFeatureValue(self):
        return self.features[23]

    def getAreaOfPolygonsFeatureValue(self):
        return self.features[24]

    def getNumberOfPolygonsFeatureValue(self):
        return self.features[25]

    def getDensityOfPolygonsFeatureValue(self):
        return self.features[26]

    def getAverageDistanceTo3NearestNeighborsFeatureValue(self):
        return self.features[27]

    def getAverageDistanceTo5NearestNeighborsFeatureValue(self):
        return self.features[28]

    def getAverageDistanceTo7NearestNeighborsFeatureValue(self):
        return self.features[29]

    def getStandardDeviationDistanceTo3NearestNeighborsFeatureValue(self):
        return self.features[30]

    def getStandardDeviationDistanceTo5NearestNeighborsFeatureValue(self):
        return self.features[31]

    def getStandardDeviationDistanceTo7NearestNeighborsFeatureValue(self):
        return self.features[32]

    def getDisorderOfDistanceTo3NearestNeighborsFeatureValue(self):
        return self.features[33]

    def getDisorderOfDistanceTo5NearestNeighborsFeatureValue(self):
        return self.features[34]

    def getDisorderOfDistanceTo7NearestNeighborsFeatureValue(self):
        return self.features[35]

    def getAvgNearestNeighborsInA10PixelRadiusFeatureValue(self):
        return self.features[36]

    def getAvgNearestNeighborsInA20PixelRadiusFeatureValue(self):
        return self.features[37]

    def getAvgNearestNeighborsInA30PixelRadiusFeatureValue(self):
        return self.features[38]

    def getAvgNearestNeighborsInA40PixelRadiusFeatureValue(self):
        return self.features[39]

    def getAvgNearestNeighborsInA50PixelRadiusFeatureValue(self):
        return self.features[40]

    def getStandardDeviationNearestNeighborsInA10PixelRadiusFeatureValue(self):
        return self.features[41]

    def getStandardDeviationNearestNeighborsInA20PixelRadiusFeatureValue(self):
        return self.features[42]

    def getStandardDeviationNearestNeighborsInA30PixelRadiusFeatureValue(self):
        return self.features[43]

    def getStandardDeviationNearestNeighborsInA40PixelRadiusFeatureValue(self):
        return self.features[44]

    def getStandardDeviationNearestNeighborsInA50PixelRadiusFeatureValue(self):
        return self.features[45]

    def getDisorderOfNearestNeighborsInA10PixelRadiusFeatureValue(self):
        return self.features[46]

    def getDisorderOfNearestNeighborsInA20PixelRadiusFeatureValue(self):
        return self.features[47]

    def getDisorderOfNearestNeighborsInA30PixelRadiusFeatureValue(self):
        return self.features[48]

    def getDisorderOfNearestNeighborsInA40PixelRadiusFeatureValue(self):
        return self.features[49]

    def getDisorderOfNearestNeighborsInA50PixelRadiusFeatureValue(self):
        return self.features[50]
