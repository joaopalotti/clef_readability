from pymetamap import MetaMap

from byeHTML import byeHTML
import glob
import gzip
from features import find_encoding
import pandas as pd
from scoop import futures

mm = MetaMap.get_instance('/bigdata/palotti/public_mm2016/bin/metamap16')

chv = pd.read_csv("../resources/chv/CHV_concepts_terms_flatfile_20110204.tsv", sep="\t",
                  names =["cui","name","pname","aname", "v", "y","n","m","a","b","v1","v2","v3","c","d"])
chv["v3"] = chv["v3"].where(chv["v3"] != "\\N", 0.0).astype(float)
chv = chv[chv["v3"] >= 0][["cui","v3"]]
chv = chv.groupby("cui")["v3"].mean().reset_index()
chv = dict(chv.values)

def process(filename):

    encoding = find_encoding(filename)

    if filename.endswith(".gz"):
        with gzip.open(filename, mode="rt", encoding=encoding, errors="surrogateescape") as f:
            content = str(f.read()) # Explicitly convert from bytes to str
    else:
        with open(filename, encoding=encoding, errors="surrogateescape", mode="r") as f:
            content = f.read()

    text = byeHTML(content, preprocesshtml="justext", forcePeriod=True).get_text()
    #print("TEXT: %s" % len(text.splitlines()))
    concepts, error = mm.extract_concepts(text.splitlines(), prefer_multiple_concepts=True, restrict_to_data_sources=["CHV"])

    n_all_concepts = 0
    n_dsyn_concepts = 0
    n_sosy_concepts = 0
    all_sum = 0
    dsyn_sum = 0
    sosy_sum = 0

    for concept in concepts:
        if concept.type != "MMI" or concept.cui not in chv:
            continue

        chv_value = chv[concept.cui]
        n_all_concepts += 1
        all_sum += chv_value

        if "dsyn" in concept.semtypes.strip("[]").split(","):
            n_dsyn_concepts += 1
            dsyn_sum += chv_value

        if "sosy" in concept.semtypes.strip("[]").split(","):
            n_sosy_concepts += 1
            sosy_sum += chv_value

    avg_chv_all =  0.0 if n_all_concepts == 0 else all_sum/n_all_concepts
    avg_chv_dsyn = 0.0 if n_dsyn_concepts == 0 else dsyn_sum/n_dsyn_concepts
    avg_chv_sosy = 0.0 if n_sosy_concepts == 0 else sosy_sum/n_sosy_concepts
    print("%s,%d,%d,%d,%.3f,%.3f,%.3f" % (filename, n_all_concepts, n_dsyn_concepts, n_sosy_concepts,
        avg_chv_all,avg_chv_dsyn,avg_chv_sosy))
    return 1

if __name__ == "__main__":

    filenames = glob.glob("../data/clef16docs/*")
    print("Result: %d" % sum(list(futures.map(process, filenames[:]))))

