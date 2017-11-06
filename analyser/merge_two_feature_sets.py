import sys
import pandas as pd
import numpy as np

features_file1 = sys.argv[1]
features_file2 = sys.argv[2]
suffix1 = sys.argv[3]
suffix2 = sys.argv[4]
output = sys.argv[5]

features1 = pd.read_csv(features_file1)
features2 = pd.read_csv(features_file2)

pd.merge(features1, features2, on="filename", suffixes=(suffix1, suffix2)).to_csv(output, index=False)

