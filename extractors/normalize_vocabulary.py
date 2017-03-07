import pandas as pd
import numpy as np
import sys

word_counts_filename = sys.argv[1]
filter_vocabulary_file = sys.argv[2] if len (sys.argv) > 2 else None

df = pd.read_csv(word_counts_filename, names=["word","freq"], dtype={'word':np.str,'freq':int})
df = df.set_index("word")
filtervoc = None

if filter_vocabulary_file:
    with open(filter_vocabulary_file) as f:
        filtervoc = set([w.strip() for w in f.readlines()] )

def normalize_vocabulary(dfi, filtered):

    if filtered:
        dfi = dfi.ix[filtered].dropna()

    dfi.dropna(inplace=True)
    dfi.sort_values(by="freq", inplace=True)
    dfi = dfi.reset_index().reset_index().rename(columns={"index":"rank"})
    dfi["rank"] = dfi["rank"] + 1
    dfi["rank"] = 100. * dfi["rank"] / dfi.shape[0]

    #dfi =  100 * (dfi - dfi.min()) / (dfi.max() - dfi.min())

    return dfi

df = normalize_vocabulary(df, filtervoc)
df[["word","rank"]].to_csv("normalized.vocabulary")


