import numpy as np
from scipy.stats import kurtosis, skew
import itertools


def nchoosek(N, k):
    comb = []
    for i in itertools.combinations(N, k):
        comb.append(list(i))
    return comb


def getFeatureStats(featVector):
    stats = []
    r = np.size(featVector)
    if r == 0:
        stats = np.full((1, 8), np.nan).tolist()
    elif np.size(featVector) == np.size(featVector, 0): #### kurtosis and std results are a little different from matlab
        stats = [np.nansum(featVector), np.nanmean(featVector), np.nanstd(featVector), np.nanmedian(featVector), np.max(featVector), np.min(featVector), kurtosis(featVector, bias=False), skew(featVector)]
    else:
        numCol = len(featVector.T)
        for i in range(numCol):
            stats.append(np.nansum(featVector.T[i]))
        for i in range(numCol):
            stats.append(np.nanmean(featVector.T[i]))
        for i in range(numCol):
            stats.append(np.nanstd(featVector.T[i]))
        for i in range(numCol):
            stats.append(np.nanmedian(featVector.T[i]))
        for i in range(numCol):
            stats.append(np.max(featVector.T[i]))
        for i in range(numCol):
            stats.append(np.min(featVector.T[i]))
        for i in range(numCol):
            stats.append(kurtosis(featVector.T[i], bias=False))
        for i in range(numCol):
            stats.append(skew(featVector.T[i]))

    return stats