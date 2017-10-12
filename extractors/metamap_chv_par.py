from pymetamap import MetaMap

import glob
import pandas as pd
from scoop import futures
import sys, os
from auxiliar import get_content

path_to_files = sys.argv[1]

mm = MetaMap.get_instance('/bigdata/palotti/public_mm2016/bin/metamap16')

chv = pd.read_csv("../resources/chv/CHV_concepts_terms_flatfile_20110204.tsv", sep="\t",
                  names =["cui","name","pname","aname", "v", "y","n","m","a","b","v1","v2","v3","c","d"])
chv["v3"] = chv["v3"].where(chv["v3"] != "\\N", 0.0).astype(float)
chv = chv[chv["v3"] >= 0][["cui","v3"]]
chv = chv.groupby("cui")["v3"].mean().reset_index()
chv = dict(chv.values)

def process(filename):

    try:
        content = get_content(filename, htmlremover=None)
        concepts, error = mm.extract_concepts(content.splitlines(), prefer_multiple_concepts=True, restrict_to_data_sources=["CHV"])
    except:
        return 0

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

    filenames = glob.glob(os.path.join(path_to_files, "*"))
    print("Result: %d" % sum(list(futures.map(process, filenames[:]))))

