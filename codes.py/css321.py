import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
import math

class Cs321(object):
    def __init__(self, ht, pep_dff):
        self.ht = ht
        self.pep_dff = pep_dff
    def hclust(self):
        ht = self.ht
        dist = pdist(self.pep_dff, metric='correlation')
        Z = hierarchy.linkage(dist, method='average')
        clusters = fcluster(Z, ht, criterion='distance')
        pd.value_counts(clusters)

        ppr = pd.value_counts(clusters) / len(clusters)

        ppr0 = np.prod(ppr[0:(len(ppr) - 1)]) * (1 - sum(ppr[0:(len(ppr) - 1)]))

        #mscore0 = math.exp(ppr0) * (math.pow(1000000,(sum(np.cumprod(ppr))-1)))
        #mscore0 = math.exp(ppr0) * (math.pow(10000000, (sum(np.cumprod(ppr)) - 1)))
        mscore0 = math.exp(ppr0) * (sum(np.cumprod(ppr)))
        mscore = mscore0 * (-math.log(ht))

        idx_str = map(str,self.pep_dff.index[clusters == ppr.index[0]])
        tgp = ';'.join(idx_str)
        rslt= list([ppr.index[0],tgp,ht,mscore,len(ppr)])

        return rslt

if __name__ == "__main__":
    Cs321()



