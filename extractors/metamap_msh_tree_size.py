from pymetamap import MetaMap

from byeHTML import byeHTML
import glob
import gzip
from features import find_encoding

filenames = glob.glob("../data/pool_2016/*")
mm = MetaMap.get_instance('/home/palotti/public_mm/bin/metamap16')

for filename in filenames[:]:
    encoding = find_encoding(filename)

    if filename.endswith(".gz"):
        with gzip.open(filename, mode="rt", encoding=encoding, errors="surrogateescape") as f:
            content = str(f.read()) # Explicitly convert from bytes to str
    else:
        with open(filename, encoding=encoding, errors="surrogateescape", mode="r") as f:
            content = f.read()

    text = byeHTML(content, preprocesshtml="justext", forcePeriod=True).get_text()
    #print("TEXT: %s" % len(text.splitlines()))
    concepts, error = mm.extract_concepts(text.splitlines(), prefer_multiple_concepts=True, restrict_to_data_sources=["MSH"])

    sum_tree_size_all = 0
    sum_tree_size_dsyn = 0
    sum_tree_size_sosy = 0
    n_all_concepts = 0
    n_dsyn_concepts = 0
    n_sosy_concepts = 0

    for concept in concepts:
        if concept.type != "MMI":
            continue

        if len(concept.tree_codes) > 0:
            n_all_concepts += 1
            sum_tree_size_all += len(concept.tree_codes.strip().split('.'))

        if "dsyn" in concept.semtypes.strip("[]").split(","):
            n_dsyn_concepts += 1
            sum_tree_size_dsyn += len(concept.tree_codes.strip().split('.'))

        if "sosy" in concept.semtypes.strip("[]").split(","):
            n_sosy_concepts += 1
            sum_tree_size_sosy += len(concept.tree_codes.strip().split('.'))

    avg_size_all =  0.0 if n_all_concepts == 0 else sum_tree_size_all/n_all_concepts
    avg_size_dsyn = 0.0 if n_dsyn_concepts == 0 else sum_tree_size_dsyn/n_dsyn_concepts
    avg_size_sosy = 0.0 if n_sosy_concepts == 0 else sum_tree_size_sosy/n_sosy_concepts
    print("%s,%d,%d,%d,%.3f,%.3f,%.3f" % (filename, n_all_concepts, n_dsyn_concepts, n_sosy_concepts,
        avg_size_all,avg_size_dsyn,avg_size_sosy))


