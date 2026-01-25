import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform


def getClosestNeigborByGroup(groups, groupId):   # dictionary including necessary cluster, clusterCentroid
    numGroups = len(groups)
    neighbors = dict()

    for j in range(numGroups):
        neighbors[j] = dict()
        if groupId == j:
            dist = squareform(pdist(groups[groupId]['clusterCentroids']))
            dist[dist == 0.0] = float('inf')
        else:
            dist = cdist(groups[groupId]['clusterCentroids'], groups[j]['clusterCentroids'])
        # got the distance matrix

        if len(groups[j]['clusters']) == 1:
            numNeigh = len(dist)
            if groupId == j:
                numNeigh = numNeigh - 1
            neighbors[j]['closestIdx'] = np.ones(numNeigh)
            neighbors[j]['sortedDist'] = dist[0:numNeigh]  # 0:numNeigh ?

        else:
            numNeigh = np.size(dist, 1)
            if groupId == j:
                numNeigh = numNeigh - 1

            idx = np.argsort(dist)
            sortedDist = np.sort(dist)
            neighbors[j]['closestIdx'] = idx.T[0:numNeigh].T
            neighbors[j]['sortedDist'] = sortedDist.T[0:numNeigh].T
    return neighbors


def ind2sub(siz, ndx):
    v1 = ndx % siz
    v2 = (ndx - v1) / siz
    v2 = [int(i) for i in v2]
    return v1, v2


def getAbsoluteClosestNeighbors(groups, groupId, maxClust):
    numGroups = len(groups)
    numClusters = groups[groupId]['numClust']
    absoluteClosest = dict()

    for j in range(numClusters):
        distMat = np.full((numGroups, maxClust), np.inf)
        for k in range(numGroups):
            if groups[groupId]['neighborsPerGroup'][k]['sortedDist'].size:
                dist = groups[groupId]['neighborsPerGroup'][k]['sortedDist'][j]
                distMat[k, 0:len(dist)] = dist

        idx = np.argsort(distMat.T.reshape(-1, 1).T[0])
        x, y = ind2sub(np.size(distMat, 0), idx)

        numIdx = np.count_nonzero(1/distMat)
        close = np.zeros((numIdx, 2))
        for k in range(numIdx):
            if np.size(groups[groupId]['neighborsPerGroup'][x[k]]['closestIdx'], 1) == 1:
                close[k] = np.array([x[k], groups[groupId]['neighborsPerGroup'][x[k]]['closestIdx'][y[k]]])
            else:
                a = x[k]
                b = y[k]
                c = groups[groupId]['neighborsPerGroup'][x[k]]['closestIdx'][j]
                d = groups[groupId]['neighborsPerGroup'][x[k]]['closestIdx'][j, y[k]]
                close[k] = [ x[k], groups[groupId]['neighborsPerGroup'][x[k]]['closestIdx'][j, y[k]] ]

        absoluteClosest[j] = dict()
        absoluteClosest[j]['idx'] = close
    return absoluteClosest
