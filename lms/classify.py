import pandas as pd
import numpy as np
import sys
import glob
import os
from auxiliar import get_words, get_content

counts_filename = sys.argv[1]
path_to_files = sys.argv[2]
files = glob.glob(os.path.join(path_to_files,"*"))
minCollectionFreq = 5
requiredToBeInNCollections = 2

def feature_selection(df, requiredToBeInNCollections, minCollectionFreq=5):

    # Remove words that appear only in one single collection:
    df["n_collections"] = df.isnull().sum(axis=1)
    df = df[ df["n_collections"] < requiredToBeInNCollections ]
    del df["n_collections"]

    # Remove words that have a collection frequency smaller than X. Default X=5
    df = df[df.sum(axis=1) > minCollectionFreq]

    return df

def smooth_normalize(df, fields, smoothing="laplace"):

    smoothfactor = {}
    for f in fields:
        smoothfactor[f] = 0

    if smoothing == "laplace":
        for f in fields:
            # Count number of non-zero terms
            smoothfactor[f] = sum(~df[f].isnull())

    oov = {}
    for f in fields:
        df[f] = np.log( (df[f] + 1.) /  (df[f].sum() + smoothfactor[f]) )
        oov[f] = 1. / (df[f].sum() + smoothfactor[f])
    return df, pd.Series(oov)


def classify_text(df, oov, words):
    dfi = df.set_index("word")

    step = 1000
    partialSum = dfi.ix[words[0:step]].fillna(oov).sum()
    for start in np.arange(step, len(words), step):
        partialSum += dfi.ix[words[start:start+step]].fillna(oov).sum()
    return partialSum.values

    #lms = dfi.ix[words].fillna(oov).sum().values
    #return lms

df = pd.read_csv(counts_filename)

freq_fields = [k for k in df.keys() if "freq" in k]
df = feature_selection(df, requiredToBeInNCollections, minCollectionFreq)
df, oov = smooth_normalize(df, freq_fields, smoothing="laplace")


for f in files:
    text = get_content(f)
    words = get_words(text)
    scores = classify_text(df, oov, words)
    print(",".join([os.path.basename(f)] + ['{:.3f}'.format(i) for i in scores]))


#df.to_csv("merged.norm", index=False)


