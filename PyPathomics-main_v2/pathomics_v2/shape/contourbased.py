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

import os

from PIL import Image
# Read Images
import numpy as np

import matplotlib.pyplot as plt
import skimage
from skimage import measure
from skimage import morphology

from re import X
from turtle import distance
from uuid import RESERVED_FUTURE
import numpy as np
import random
import os
# from scipy import io
import scipy
from scipy import spatial
import scipy.io as sio
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import sklearn
from sklearn import decomposition
from operator import itemgetter


def centroid(xy):
    # % *****>      Input:
    # %               XY, an n-by-2 matrix whose rows are
    # %             counterclockwise listed coordinates of the
    # %             vertices.
    # % *****>      Output: CX, CY - coordinates of the centroid
    # %                          A - area of the region
    # % *****>      Call: [CX CY A]=centroid(XY)
    # %
    # %             Method: (position)=(static moment)/(area)
    # %                    transformed using Green's formula
    # %             (x-moment)=(area integral)(x dA)
    # %                       =(boundary integral)(-xy dx)
    # %             similarly (up to sign) for y-moment;
    # %             (area)=(1/2)*(boundary integral)(x dy - y dx)
    # %             evaluated along the boundary edges and
    # %             re-arranged for greater symmetry
    # %             By Z.V. Kovarik (kovarik@mcmaster.ca), May 1996
    ST = xy[1:] + [xy[0]]  ## cyclic shift of XY
    UV = np.add(ST, xy)  ## element-wise addition
    ST = np.subtract(ST, xy)

    CXY = (3 * UV[:, 0] * UV[:, 1] + ST[:, 0] * ST[:, 1]) @ ST / 12
    A = np.sum(UV[:, 0] * ST[:, 1] - UV[:, 1] * ST[:, 0]) / 4
    CXY = CXY / A
    CX = -CXY[0]
    CY = CXY[1]  ##centroid coordinates
    A = np.abs(A)
    return CX, CY, A


def distratio(xy):
    # # %Program for new distance ratio
    # # %xy being the matrix containin x and y coordinates
    # xy = np.array(xy)
    # n = xy.shape[0]
    n = len(xy)
    if n >= 800:
        k = n / 200
        j = np.floor(k) * 200
        j = j - 199 if np.abs(n - j) == 0 else j + 1
        s = (j - 1) / 200
        lxy = []
        for i in range(int(s) + 1):  ##i=1:s+1
            lxy.append(xy[i * 200])

    if n > 400 and n < 800:
        k = n / 100
        j = np.floor(k) * 100
        j = j - 99 if np.abs(n - j) == 0 else j + 1
        s = (j - 1) / 100
        lxy = []
        for i in range(int(s) + 1):  ##i=1:s+1
            lxy.append(xy[i * 100])

    if n <= 400:
        k = n / 10
        j = np.floor(k) * 5
        j = j + 51 if np.abs(n - j) >= 51 else j + 1
        s = (j - 1) / 5
        lxy = []
        for i in range(int(s) + 1):  ##i=1:s+1
            lxy.append(xy[i * 5])
    dislong = []
    for a in range(len(lxy) - 1):
        dislong.append(np.linalg.norm(np.array(lxy[a]) - np.array(lxy[a + 1])))
    disshort = []
    for a in range(int(j) - 1):
        disshort.append(np.linalg.norm(np.array(xy[a]) - np.array(xy[a + 1])))
    dratio = np.sum(dislong) / np.sum(disshort) if np.sum(disshort) != 0 else 0.0
    return dratio


def periarea(xy, area):
    n = len(xy)
    dist = []
    for i in range(n - 1):
        dist.append(np.linalg.norm(np.array(xy[i]) - np.array(xy[i + 1])))
    ld = np.linalg.norm(np.array(xy[n - 1]) - np.array(xy[0]))
    peri = np.sum(dist) + ld
    paratio = (peri * peri) / area
    return paratio, peri


def frdescp(s_):
    # %FRDESCP Computes fourier descriptors.
    # % Z=FRDESCP(S) computes the Fourier descriptors of S, which is an np by 2
    # % sequence o image coordinates describing a boundary

    # s = np.array(s)
    # [nx,ny] = s.shape
    s = s_.copy()
    nx = len(s)
    ny = len(s[0])
    if nx / 2 != round(nx / 2):
        s.append(s[nx - 1])
        nx = len(s)

    ## %create an alternating sequence of 1s and -1s for use in centering the transform
    m = [(-1)**x for x in range(nx)]
    s = np.array(s)
    ## %Multiply the input sequence by alternating 1s and -1s to center the transform.
    s[:, 0] = np.array(m) * s[:, 0]
    s[:, 1] = np.array(m) * s[:, 1]
    s_complex = s[:,
                  0] + s[:, 1] * 1j  ##%convert coordinates to complex numbers.
    # s_complex = np.array([s[ii,0] + s[ii,1]*1j for ii in range(len(s))])
    z = np.real(np.fft.fft(s_complex))  ##%compute the descriptors
    return z


def bound2im(b):
    # %Bound2IM converts a boundary to an image.
    # %   B=bound2im(b) converts b, an npx2 or 2xnp into a binary image with 1s
    # %   in the locations defined by the coordinates in b and 0s elsewhere.

    b = np.array(b)
    x = np.round(b[:, 0])
    y = np.round(b[:, 1])
    x, y = np.int64(x), np.int64(y)
    x = x - np.min(x)
    y = y - np.min(y)
    B = np.full((np.max(x) + 1, np.max(y) + 1), False)
    C = np.max(x) - np.min(x)
    D = np.max(y) - np.min(y)
    B[x, y] = True
    return B


def compute_phi(e):
    phi_1 = e['eta20'] + e['eta02']
    phi_2 = (e['eta20'] - e['eta02'])**2 + 4 * e['eta11']**2
    phi_3 = (e['eta30'] - 3 * e['eta12'])**2 + (3 * e['eta21'] - e['eta03'])**2
    phi_4 = (e['eta30'] + e['eta12'])**2 + (e['eta21'] + e['eta03'])**2
    phi_5 = (e['eta30'] - 3 * e['eta12']) * (e['eta30'] + e['eta12']) * (
        (e['eta30'] + e['eta12'])**2 - 3 *
        (e['eta21'] + e['eta03'])**2) + (3 * e['eta21'] - e['eta03']) * (
            e['eta21'] + e['eta03']) * (3 * (e['eta30'] + e['eta12'])**2 -
                                        (e['eta21'] + e['eta03'])**2)
    phi_6 = (e['eta20'] - e['eta02']) * (
        (e['eta30'] + e['eta12'])**2 -
        (e['eta21'] + e['eta03'])**2) + 4 * e['eta11'] * (
            e['eta30'] + e['eta12']) * (e['eta21'] + e['eta03'])
    phi_7 = (3 * e['eta21'] - e['eta03']) * (e['eta30'] + e['eta12']) * (
        (e['eta30'] + e['eta12'])**2 - 3 *
        (e['eta21'] + e['eta03'])**2) + (3 * e['eta12'] - e['eta30']) * (
            e['eta21'] + e['eta03']) * (3 * (e['eta30'] + e['eta12'])**2 -
                                        (e['eta21'] + e['eta03'])**2)
    # phi = [phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7]
    return phi_1, phi_2, phi_3, phi_4, phi_5, phi_6, phi_7


def compute_eta(m):
    xbar = m['m10'] / m['m00']
    ybar = m['m01'] / m['m00']

    e = {}
    e['eta11'] = (m['m11'] - ybar * m['m10']) / m['m00']**2
    e['eta20'] = (m['m20'] - xbar * m['m10']) / m['m00']**2
    e['eta02'] = (m['m02'] - ybar * m['m01']) / m['m00']**2
    e['eta30'] = (m['m30'] - 3 * xbar * m['m20'] +
                  2 * xbar**2 * m['m10']) / m['m00']**2.5
    e['eta03'] = (m['m03'] - 3 * ybar * m['m02'] +
                  2 * ybar**2 * m['m01']) / m['m00']**2.5
    e['eta21'] = (m['m21'] - 2 * xbar * m['m11'] - ybar * m['m20'] +
                  2 * xbar**2 * m['m01']) / m['m00']**2.5
    e['eta12'] = (m['m12'] - 2 * ybar * m['m11'] - xbar * m['m02'] +
                  2 * ybar**2 * m['m10']) / m['m00']**2.5
    return e


def compute_m(F):
    F = F * 1.0
    M, N = F.shape
    x, y = np.meshgrid(range(1, N + 1), range(
        1,
        M + 1))  ###matlab:1:N  python:range(1, N+1)  think about if range(N)
    m = {}
    m['m00'] = np.sum(F) if np.sum(F) != 0 else np.finfo(float).eps
    m['m10'] = np.sum(x * F)
    m['m01'] = np.sum(y * F)
    m['m11'] = np.sum(x * y * F)
    m['m20'] = np.sum(x * x * F)
    m['m02'] = np.sum(y * y * F)
    m['m30'] = np.sum(x * x * x * F)
    m['m03'] = np.sum(y * y * y * F)
    m['m12'] = np.sum(x * y * y * F)
    m['m21'] = np.sum(x * x * y * F)
    return m


def invmoments(F):
    phi = compute_phi(compute_eta(compute_m(F)))
    return phi


def fractal_dim(xy):
    ## %fractal dimension
    ## %xy being the matrix containin x and y coordinates
    n = len(xy)
    d = []
    for i in range(int(n / 2)):
        j = int(n / (i + 1))
        lxy = []
        for s in range(j):
            lxy.append(xy[s * (i + 1)])
        lxy.append(xy[-1])

        ## %for long distances
        dislong = []
        lxy_ = np.array(lxy)
        for a in range(len(lxy) - 1):
            dislong.append(np.linalg.norm(lxy_[a, :] - lxy_[a + 1]))
        d.append(np.sum(np.array(dislong)))
    eq = np.polyfit(np.log10(1 / np.array(range(1,
                                                int(n / 2) + 1))),
                    np.log10(np.array(d)), 1)
    frac_dim = eq[0]
    return frac_dim


class PathomicsContourBased(base.PathomicsFeaturesBase):

    def __init__(self, inputImage, inputMask, **kwargs):
        super(PathomicsContourBased, self).__init__(inputImage, inputMask, **kwargs)
        self.name = 'ContourBased'
        self.image = sitk.GetArrayViewFromImage(self.inputImage)
        self.mask = sitk.GetArrayViewFromImage(self.inputMask)
        if len(self.mask.shape) > 2:
            self.mask = self.mask[:, :, 0]
        self.bounds, self.image_intensity, self.feats = self.mask2bounds(
            self.mask, self.image, atts=['area'], **kwargs)
        self.xs = []
        self.ys = []
        self.x_coords = []
        self.y_coords = []
        self.xys = []
        self.xcs = []
        self.ycs = []
        self.areas = []
        self.distances = []
        self.dist_mins = []
        self.dist_maxs = []

        self.ims = []
        self.fds = []
        
        for b in self.bounds:
            x = b[0]
            if len(x) < 3:
                continue
            y = b[1]
            x_coord = b[4]
            y_coord = b[5]
            xy = list(map(list, zip(x,y)))
            xy = xy[::1]
            xc, yc, area = centroid(xy)
            distance = np.linalg.norm(np.array(xy) - np.array([xc, yc]), axis=1)
            dist_min, dist_max = np.min(distance), np.max(distance)

            self.xs.append(x)
            self.ys.append(y)
            self.x_coords.append(x_coord)
            self.y_coords.append(y_coord)
            self.xys.append(xy)
            self.xcs.append(xc)
            self.ycs.append(yc)
            self.areas.append(area)
            self.distances.append(distance)
            self.dist_mins.append(dist_min)
            self.dist_maxs.append(dist_max)

            B = bound2im(xy)
            phi = invmoments(B)
            self.ims.append(phi)

            z = frdescp(xy)
            fd_pre = z[:10]
            fd = np.zeros([10,])
            fd[:len(fd_pre)] = fd_pre
            self.fds.append(fd)
        
        self.ims = np.array(self.ims)
        self.fds = np.array(self.fds)




    def getAreaRatioFeatureValue(self):
        """
        1. Area Ratio
        """
        res = []
        for i in range(len(self.xs)):
            max_area = np.pi * self.dist_maxs[i] * self.dist_maxs[i]
            area = self.areas[i]
            res.append(area / max_area)
        return np.array(res)

    def getDistanceRatioFeatureValue(self):
        """
        2. Distance Ratio
        """
        res = []
        for i in range(len(self.xs)):
            dist_mean = np.mean(self.distances[i])
            dist_max = self.dist_maxs[i]
            res.append(dist_mean / dist_max)
        return np.array(res)

    def getStdDistanceFeatureValue(self):
        """
        3. Standard Deviation of Distance
        """
        res = []
        for i in range(len(self.xs)):
            dists = self.distances[i] / self.dist_maxs[i]
            dists_std = np.std(dists, ddof=1)
            res.append(dists_std)
        return np.array(res)

    def getVarianceDistanceFeatureValue(self):
        """
        4. Variance of Distance
        """
        res = []
        for i in range(len(self.xs)):
            dists = self.distances[i] / self.dist_maxs[i]
            dists_var = np.cov(dists)
            res.append(dists_var.tolist())
        return np.array(res)

    def getLongOverShortDistanceRatioFeatureValue(self):
        """
        5. Long Over Short (Long/Short) Distance Ratio
        """
        res = []
        for i in range(len(self.xs)):
            xy = self.xys[i]
            dratio = distratio(xy)
            res.append(dratio)
        return np.array(res)

    def getPerimeterRatioFeatureValue(self):
        """
        6. Perimeter Ratio
        """
        res = []
        for i in range(len(self.xs)):
            xy = self.xys[i]
            area = self.areas[i]
            paratio, _ = periarea(xy, area)
            res.append(paratio)
        return np.array(res)

    def getSmoothnessFeatureValue(self):
        """
        7. Smoothness
        """
        res = []
        for _i in range(len(self.xs)):
            D = self.distances[_i]
            s = len(D)
            sm = []
            for i in range(s):
                if i == 0:
                    sm.append(np.abs(D[i] - (D[i + 1] + D[s - 1]) / 2))
                elif i == s - 1:
                    sm.append(np.abs(D[i] - (D[0] + D[s - 2]) / 2))
                else:
                    sm.append(np.abs(D[i] - (D[i + 1] + D[i - 1]) / 2))
            res.append(np.sum(sm))   
        return np.array(res)

    def getInvariantMoment1FeatureValue(self):
        """
        8. Invariant Moment 1
        """
        return self.ims[:, 0]
    def getInvariantMoment2FeatureValue(self):
        """
        9. Invariant Moment 2
        """
        return self.ims[:, 1]
    
    def getInvariantMoment3FeatureValue(self):
        """
        10. Invariant Moment 3
        """
        return self.ims[:, 2]
    
    def getInvariantMoment4FeatureValue(self):
        """
        11. Invariant Moment 4
        """
        return self.ims[:, 3]
    
    def getInvariantMoment5FeatureValue(self):
        """
        12. Invariant Moment 5
        """
        return self.ims[:, 4]
    
    def getInvariantMoment6FeatureValue(self):
        """
        13. Invariant Moment 6
        """
        return self.ims[:, 5]
    
    def getInvariantMoment7FeatureValue(self):
        """
        14. Invariant Moment 7
        """
        return self.ims[:, 6]

    # def getFractalDimensionFeatureValue(self):
    #     """
    #     15. Fractal Dimension
    #     """
    #     res = []
    #     for i in range(len(self.xs)):
    #         xy = self.xys[i]
    #         frac_dim = fractal_dim(xy)
    #         res.append(frac_dim)
    #     return np.array(res)

    def getFourierDescriptor1FeatureValue(self):
        """
        16. Fourier Descriptor 1
        """
        return self.fds[:, 0]

    def getFourierDescriptor2FeatureValue(self):
        """
        17. Fourier Descriptor 2
        """
        return self.fds[:, 1]
    
    def getFourierDescriptor3FeatureValue(self):
        """
        18. Fourier Descriptor 3
        """
        return self.fds[:, 2]
    
    def getFourierDescriptor4FeatureValue(self):
        """
        19. Fourier Descriptor 4
        """
        return self.fds[:, 3]
    
    def getFourierDescriptor5FeatureValue(self):
        """
        20. Fourier Descriptor 5
        """
        return self.fds[:, 4]
    
    def getFourierDescriptor6FeatureValue(self):
        """
        21. Fourier Descriptor 6
        """
        return self.fds[:, 5]
    
    def getFourierDescriptor7FeatureValue(self):
        """
        22. Fourier Descriptor 7
        """
        return self.fds[:, 6]
    
    def getFourierDescriptor8FeatureValue(self):
        """
        23. Fourier Descriptor 8
        """
        return self.fds[:, 7]
    
    def getFourierDescriptor9FeatureValue(self):
        """
        24. Fourier Descriptor 9
        """
        return self.fds[:, 8]
    
    def getFourierDescriptor10FeatureValue(self):
        """
        25. Fourier Descriptor 10
        """
        return self.fds[:, 9]