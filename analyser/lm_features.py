import numpy as np
import pandas as pd


"""
Features based on clef15 were used to generate clef2016.lms
"""
clef_lms = pd.read_csv("clef2016_lm_from_clef2015.txt")
clef_lms.sort_values(by="filename", inplace=True)

"""
Features based on simple/english wikipedia were used to generate wiki_lm.csv
"""
wiki_lms = pd.read_csv("clef2016_lm_from_wiki.txt")
wiki_lms.sort_values(by="filename", inplace=True)

lms = pd.merge(wiki_lms, clef_lms, on="filename")

textfeatures = pd.read_csv("text_features_allprocessing.csv")

def greatest(values):
    max_i = np.argmax(values.values)
    ans = np.zeros(len(values))
    ans[max_i] = 1.
    return ans

greatest_clef = lms[["lm_clef15_model0", "lm_clef15_model1","lm_clef15_model2","lm_clef15_model3"]].apply(greatest, axis=1)
greatest_clef = greatest_clef.rename(columns={"lm_clef15_model0":"lm_clef15_big0", "lm_clef15_model1":"lm_clef15_big1","lm_clef15_model2":"lm_clef15_big2", "lm_clef15_model3":"lm_clef15_big3"})

# Actually I should not use two variables here. Instead, I should have used something like simple>eng?
greatest_wiki = lms[["simplewiki","enwiki"]].apply(greatest, axis=1)
greatest_wiki = greatest_wiki.rename(columns={"simplewiki":"simplewiki_greatest", "enwiki":"enwiki_greatest"})

lms = pd.concat((lms, greatest_clef, greatest_wiki), axis=1)

merged = pd.merge(textfeatures, lms)
merged.to_csv("text_lm_features.csv", index=False)

