import pandas as pd
import sys

fout = sys.argv[1]
inputs = sys.argv[2:]

df = pd.read_csv(inputs[0], names=["word", "freq0"])
print("Processed: %s" % (inputs[0]))


for i in range(len(inputs)-1):
    other = inputs[i+1]
    odf = pd.read_csv(other, names=["word", "freq%d" % (i+1)])
    df = pd.merge(df, odf, how="outer", on="word")
    print("Processed: %s" % (other))

df.to_csv(fout, index=False)
print("DONE! Created %s" % (fout))

