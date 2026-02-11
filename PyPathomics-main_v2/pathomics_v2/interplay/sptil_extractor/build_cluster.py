import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull
import math


def getCCGEdges(nodes, alpha, r):
    D = squareform(pdist(nodes, 'euclidean'))
    # P = math.pow(D, -alpha)
    # P = D ** -alpha
    P = np.power(D+1e-9, -alpha)
    edges = np.triu(P > r, 1)
    return edges


def networkComponments(edges):
    N = len(edges)
    for i in range(N):
        edges[i][i] = 0
    edges = edges + edges.T
    isDiscovered = np.zeros(N)

    # members = dict()
    mbs = []
    member = []
    num_of_clt = -1   # make the first cluster index = 0
    for n in range(N):
        num_of_clt +=1  # next cluster
        if not isDiscovered[n]: # vist the cell node that haven't yet
            # members[num_of_clt] = []  # prepare a new list to contain cell nodes
            # members[num_of_clt].append(n)
            member.append(n)
            isDiscovered[n] = 1  # mark this  stem cell as visted
            idx_of_clt_lst = 0     # problem: after finishing the first cluster, it run out of the circle
            # while idx_of_clt_lst < len(members[num_of_clt]):
            while idx_of_clt_lst < len(member):
                nbrs = np.where(edges[member[idx_of_clt_lst]])[0]  # find neighbors of the stem cell node    # nbrs = np.where(edges[members[num_of_clt][idx_of_clt_lst]])[0]
                newNbrs = nbrs[np.where(isDiscovered[nbrs] == 0)[0]]  # find the nodes that haven't presented
                isDiscovered[newNbrs] = 1  # mark them as discovered
                for t in range(len(newNbrs)):
                    # members[num_of_clt].append(newNbrs[t])
                    member.append(newNbrs[t])
                idx_of_clt_lst += 1
            mbs.append(member)
            member = []
        else:
            num_of_clt -= 1

    mbs = sorted(mbs, key=len, reverse=True)
    for num_of_clt in range(len(mbs)):
        idx_of_sm_clt = 0
        if len(mbs[num_of_clt]) <= 2:
            idx_of_sm_clt = num_of_clt
            break
    del mbs[idx_of_sm_clt:len(mbs)]

    nComponents = len(mbs)
    sizes = np.zeros(nComponents)
    for num_of_clt in range(nComponents):
        sizes[num_of_clt] = len(mbs[num_of_clt])

    idx = np.argsort(-sizes)  # decending sort index
    sizes = sizes[idx]

    return nComponents, sizes, mbs


def getClusterProperties(clusters, nodes):  # coords_0,1  dateframe      nodes = np.array(coords)
    num_of_clt = len(clusters)
    centroids = np.zeros(num_of_clt*2).reshape(-1, 2)
    polygons = []
    plgn = []
    areas = np.zeros(num_of_clt)
    densities = np.zeros(num_of_clt)

    for i in range(num_of_clt):
        member = clusters[i]
        # col = coords.iloc[member, 0]
        # row = coords.iloc[member, 1]
        points = nodes[member]
        hull = ConvexHull(points)

        areas[i] = hull.volume
        densities[i] = len(points)/hull.volume

        '''
        testx = points[hull.vertices, 0].tolist()
        testx.append(testx[1])
        testy = points[hull.vertices, 1].tolist()
        testy.append(testy[1])
        cx = np.mean(testx)
        cy = np.mean(testy)
        '''

        cx = np.mean(points[hull.vertices, 0])
        cy = np.mean(points[hull.vertices, 1])
        centroids[i] = [cx, cy]
        plgn.append(points[hull.vertices])
        polygons.append(plgn[0])
        plgn = []
    return centroids, polygons, areas, densities
