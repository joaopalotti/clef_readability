import pandas as pd
import re

concepts = pd.read_csv("CHV_concepts_terms_flatfile_20110204.tsv", sep="\t", names=["cui","name","pname","aname", "v", "y","n","m","a","b","v1","v2","v3","c","d"])

concepts["v3"] = concepts["v3"].where(concepts["v3"] != "\N", 0.0).astype(float)
concepts["name"] = concepts["name"].astype(str)
concepts["name"] = concepts["name"].apply(lambda x: re.sub(r'\([^)]*\)', '', x))
concepts["name"] = concepts["name"].apply(lambda x: re.sub(r'[^0-9a-zA-Z]+', ' ', x))
concepts["name"] = concepts["name"].apply(lambda x: ' '.join(x.split()) )
concepts["name"] = concepts["name"].apply(lambda x: x.lower() )

# remove -1's
concepts = concepts[concepts["v3"] >= 0]
concepts[["name","v3"]].to_csv("chv_concepts.csv", sep=",", index=False, header=0)

