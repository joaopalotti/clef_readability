import pandas as pd
import sys
from sklearn.preprocessing import OneHotEncoder

filein = sys.argv[1]


def label(row, mapping):
    m = row[mapping.keys()].argmax()
    return mapping[m]

df = pd.read_csv(filein)

#df = df.set_index("filename")

mapping = {"LMVeryHard":0,"LMHard":1,"LMEasy":2,"LMEasiest":3}
df["chosen"] = df.apply(lambda x: label(x, mapping), axis=1)

enc = OneHotEncoder()
m = enc.fit_transform(df["chosen"].values.reshape(-1,1)).todense().astype(int)

df = pd.concat((df,pd.DataFrame(m)), axis=1)
df = df.rename(columns={0:"isVeryHard",1:"isHard",2:"isEasy",3:"isVeryEasy"})
df.to_csv("clef2016_lm15.txt", index=False)


