import pandas as pd
import numpy as np
import sys
import glob
import os
from auxiliar import get_words, get_content

vocabulary_filename = sys.argv[1]
path_to_files = sys.argv[2]

files = glob.glob(os.path.join(path_to_files,"*"))

def score_text(df, words):
    dfi = df.set_index("word")

    step = 1000


    values = dfi.ix[words[0:step]]["rank"].values
    for start in np.arange(step, len(words), step):
        values = np.concatenate((values, dfi.ix[words[start:start+step]]["rank"].values))

    if values.shape[0] == 0:
        return (0.,0.,0.,0.,0.,0.,0.)


    return (np.nanmin(values), np.nanpercentile(values, 25), np.nanpercentile(values, 50), np.nanpercentile(values, 75), np.nanmax(values), np.nanmean(values), np.nan_to_num(values).mean())

df = pd.read_csv(vocabulary_filename, names=["word","rank"])

for f in files:
    text = get_content(f)
    words = get_words(text)
    scores = score_text(df, words)
    scores = ",".join(map("{:.3f}".format,list(scores)))
    print("%s,%s" % (os.path.basename(f), scores))


#df.to_csv("merged.norm", index=False)
