import pandas as pd
import sys

makeone = True

inputs = sys.argv[1:]

df = pd.read_csv(inputs[0], names=["word", "freq0"])
print("Processed: %s" % (inputs[0]))


for i in range(len(inputs)-1):
    other = inputs[i+1]
    odf = pd.read_csv(other, names=["word", "freq%d" % (i+1)])
    df = pd.merge(df, odf, how="outer", on="word")
    print("Processed: %s" % (other))


if makeone:
    df.set_index("word").sum(axis=1).to_csv(sys.stdout)
else:
    df.fillna(0.0).to_csv(sys.stdout, index=False)

print("DONE!")

