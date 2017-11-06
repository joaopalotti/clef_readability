import pandas as pd
import numpy as np
import sys
import glob
import os
from auxiliar import get_words, get_content
from scoop import futures

vocabulary_filename = sys.argv[1]
path_to_files = sys.argv[2]

df = pd.read_csv(vocabulary_filename)
dfi = df.set_index("word")

def score_text(dfi, words):

    step = 1000

    try:
        values = dfi.loc[words[0:step]]["rank"].values
        for start in np.arange(step, len(words), step):
            values = np.concatenate((values, dfi.loc[words[start:start+step]]["rank"].values))

    except:
        return (0.,0.,0.,0.,0.,0.,0.)

    if values.shape[0] == 0:
        return (0.,0.,0.,0.,0.,0.,0.)

    return (np.nanmin(values), np.nanpercentile(values, 25), np.nanpercentile(values, 50), np.nanpercentile(values, 75), np.nanmax(values), np.nanmean(values), np.nan_to_num(values).mean())

def go(filename):
    text = get_content(filename, htmlremover=None)
    words = get_words(text)
    scores = score_text(dfi, words)
    scores = ",".join(map("{:.3f}".format,list(scores)))
    return (os.path.basename(filename), scores)

if __name__ == "__main__":
    files = glob.glob(os.path.join(path_to_files,"*"))
    result = futures.map(go,files)
    for r in result:
        print ("%s,%s" % r)


