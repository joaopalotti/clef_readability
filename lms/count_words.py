import sys
import pandas as pd
from auxiliar import get_words
from collections import defaultdict

fname = sys.argv[1]
fout = sys.argv[2]

vocabulary = defaultdict(int)

with open(fname, "r") as f:
    for line in f.readlines():
        for w in get_words(line):
            vocabulary[w] += 1

pd.Series(vocabulary).to_csv(fout)
