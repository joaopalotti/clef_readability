import sys
import pandas as pd
import numpy as np
from auxiliar import applyUnderstandabilityMap, normalizeFeatures, removeEmpties, clip_readability, removeDuplicates
from itertools import permutations
from multiprocessing import Pool

features_file = sys.argv[1]

qrels = pd.read_csv("../qrels/clef16/doc_all.mean")
doc_features = pd.read_csv(features_file)

# preprocess datasets:
applyUnderstandabilityMap(qrels)
doc_features = normalizeFeatures(doc_features)
doc_features = clip_readability(doc_features)

merged = pd.merge(qrels, doc_features)

relevant = merged[merged["rel"] > 0]
relevant = removeEmpties(relevant)

# Remove duplicated cols
relevant = removeDuplicates(relevant)

labels = relevant["classunders"].sort_values()

remove_keys = ['filename', 'rel', 'trust', 'unders', 'classunders']
valid_features = list(set(relevant.keys()) - set(remove_keys))

X = relevant[valid_features]
Xnona = X.fillna(0.0)

def goodman_kruskal_gamma(m, n):
    """
    compute the Goodman and Kruskal gamma rank correlation coefficient;
    this statistic ignores ties is unsuitable when the number of ties in the
    data is high. it's also slow.
    >> x = [2, 8, 5, 4, 2, 6, 1, 4, 5, 7, 4]
    >> y = [3, 9, 4, 3, 1, 7, 2, 5, 6, 8, 3]
    >> goodman_kruskal_gamma(x, y)
    0.9166666666666666
    from https://github.com/shilad/context-sensitive-sr/blob/master/SRSurvey/src/python/correlation.py
    """
    if len(m)==0  or len(n) == 0:
        print("ERROR!!")
        return 0.0

    if min(n) == max(n) or min(m) == max(m):
        print("No variance!!! ERROR!!")
        return 0.0

    num = 0
    den = 0
    for (i, j) in permutations(list(range(len(m))), 2):
        m_dir = m[i] - m[j]
        n_dir = n[i] - n[j]
        sign = m_dir * n_dir
        if sign > 0:
            num += 1
            den += 1
        elif sign < 0:
            num -= 1
            den += 1
    return num / float(den)

parameters = []
metrics = []
for k in set(relevant.keys()) - set(['filename', u'rel', u'trust', u'unders']):
    parameters.append((relevant["classunders"].values, relevant[k].values))
    metrics.append(k)

def parallel_gkg(x):
    #print("Running: %s, %s" % (x[0],x[1]))
    ans = goodman_kruskal_gamma(x[0], x[1])
    return ans

p = Pool()
scores = p.map(parallel_gkg, parameters[:])
results = dict(zip(metrics,scores))

"""
# Debug:
for i, p in enumerate(parameters):
    print("Metric: %s" % (metrics[i]))
    parallel_gkg(p)
"""

def printTable(metrics):
    for (metric, score) in metrics.iteritems():
        if "bs4" in metric:
            print("Textual\tBS4\t%.3f" % (score))
        elif "jst" in metric:
            print("Textual\tJST\t%.3f" % (score))


#printTable(results)

