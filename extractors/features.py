import sys
import gzip
import re
import os
import glob
import csv
from readcalc import readcalc
from scoop import futures, shared
import numpy as np

resource_path = "../resources/"

def count_acronyms(words, acronyms):
    #print([w for w in words if w in acronyms] )
    return sum((1 for w in words if w in acronyms))

def count_prefixes(words, prefixes):
    #print([w for p in prefixes for w in words if len(w) > 3 and w.lower().startswith(p)])
    return sum((1 for p in prefixes for w in words if len(w) > 3 and w.lower().startswith(p)))

def count_sufixes(words, sufixes):
    #print([w for s in sufixes for w in words if len(w) > 3 and w.lower().endswith(s)])
    return sum((1 for s in sufixes for w in words if len(w) > 3 and w.lower().endswith(s)))

def count_numbers(words):
    #print([w for w in words if w.isdigit()])
    return sum((1 for w in words if w.isdigit()))

def count_dict_words(words, dic, min_size=0):
    #print([w for w in words if w.lower() in dic])
    return sum((1 for w in words if w.lower() in dic if len(w) > min_size))

def calc_chv(words, chv_map):
    text = ' '.join(words)
    chvs = [v for (w,v) in chv_map if " " + w + " " in text]
    if len(chvs) == 0:
        return 0,0.0,0.0
    return len(chvs), np.mean(chvs), np.sum(chvs)

def find_encoding(doc_full_path):
    # An alternative is using a package such as chardet:
    # http://chardet.readthedocs.io/en/latest/usage.html

    encoding = "windows-1252"

    if doc_full_path.endswith(".gz"):
        f = gzip.open(doc_full_path)
    else:
        f = open(doc_full_path)

    content = f.read()[0:5000]
    regexp = re.search('charset=(?P<enc>[\s"\']*([^\s"\'/>]*))', str(content))
    if regexp is not None:
        encoding = regexp.group("enc")
    f.close()
    return encoding

def process(filename):

    continuous = shared.getConst('continuous')
    preprocessing = shared.getConst('preprocessing')

    outdir = shared.getConst('outdir')

    prefixes = shared.getConst('prefixes')
    sufixes = shared.getConst('sufixes')
    acronyms = shared.getConst('acronyms')
    eng_dict = shared.getConst('eng_dict')
    mesh_dict = shared.getConst('mesh_dict')
    stopwords_dict = shared.getConst('stopwords_dict')
    drugbank_dict = shared.getConst('drugbank_dict')
    icd_dict = shared.getConst('icd_dict')
    chv_map = shared.getConst('chv_map')

    outfilename = outdir + "/" + filename.split("/")[-1]
    csv_file = open(outfilename, 'w')
    csv_writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)

    encoding = find_encoding(filename).strip("\"\'")

    print(("Processing: %s. Encoding: %s" % (filename, encoding)))

    if filename.endswith(".gz"):
        with gzip.open(filename, mode="rb") as f:
            content = f.read()
    else:
        with open(filename) as f:
            content = f.read()

    calc = readcalc.ReadCalc(content, preprocesshtml=preprocessing)
    # print("#Words after preprocessing: %d" % (len(calc.get_words())))

    prefixes_found = count_prefixes(calc.get_words(), prefixes)
    sufixes_found = count_sufixes(calc.get_words(), sufixes)
    acronyms_found = count_acronyms(calc.get_words(), acronyms)
    numbers_found = count_numbers(calc.get_words())
    eng_found = count_dict_words(calc.get_words(), eng_dict)
    mesh_found = count_dict_words(calc.get_words(), mesh_dict, 3)
    stopwords_found = count_dict_words(calc.get_words(), stopwords_dict)
    drugbank_found = count_dict_words(calc.get_words(), drugbank_dict, 3)
    icd_found = count_dict_words(calc.get_words(), icd_dict, 3)
    chv_num, chv_mean, chv_sum = calc_chv(calc.get_words(), chv_map)

    filename = os.path.basename(filename)

    readability_row = [filename] + list(calc.get_all_metrics())
    readability_row.extend( [prefixes_found, sufixes_found, acronyms_found, numbers_found, eng_found, mesh_found,\
            stopwords_found, drugbank_found, icd_found, chv_num, chv_mean, chv_sum] )

    # Header: filename, number_chars, number_words, number_sentences, number_syllables, number_polysyllable_words, difficult_words, longer_4, longer_6, longer_10, longer_13, flesch_reading_ease, flesch_kincaid_grade_level, coleman_liau_index, gunning_fog_index, smog_index, ari_index, lix_index, dale_chall_score, prefixes_found, sufixes_found, acronyms_found, numbers_found, eng_found, mesh_found, stopwords_found, drugbank_found, icd_found, chv_num, chv_mean, chv_sum

    csv_writer.writerow(readability_row)
    csv_file.flush()
    csv_file.close()
    print(("File %s saved." % (outfilename)))
    return 1

## ========================================= ##
## ---------------Main---------------------- ##
## ========================================= ##

if __name__ == "__main__":

    if len(sys.argv) <= 1:
        script_name = os.path.basename(__file__)
        print(("USAGE: python %s [-f] <PATH_TO_DATA> <OUT_DIR>" % (script_name)))
        sys.exit(0)

    continuous=True            # Options: True, False
    preprocessing = "justext" # Options: justext, bs4

    print(("PARAMETERS: ", sys.argv))
    path_to_data = sys.argv[1]
    outdir = sys.argv[2]
    print(("Continuoes: %s, Preprocessing: %s" % (continuous, preprocessing)))

    sufix_file = os.path.join(resource_path, "suffixes")
    prefix_file = os.path.join(resource_path, "prefixes")
    acronyms_file = os.path.join(resource_path, "acronyms.txt")

    stopwords_english_dict_file = os.path.join(resource_path, "stopwords.txt")
    general_english_dict_file = os.path.join(resource_path, "en-merged.dic")
    mesh_dict_file = os.path.join(resource_path, "mesh.bag")
    drugbank_dict_file = os.path.join(resource_path, "drugbank.bag")
    icd_dict_file = os.path.join(resource_path, "icd.bag")
    chv_dict_file = os.path.join(resource_path, "chv_concepts.csv")

    sufixes = set([f.strip() for f in open(sufix_file, "r", encoding="utf8").readlines()])
    sufixes = set([f for f in sufixes if len(f) >= 4])
    prefixes = set([f.strip() for f in open(prefix_file, "r", encoding="utf8").readlines()])
    prefixes = set([f for f in prefixes if len(f) >= 4])
    acronyms = set([f.strip() for f in open(acronyms_file, "r", encoding="utf8").readlines()])

    eng_dict = set([f.strip().lower() for f in open(general_english_dict_file, "r", encoding="utf8").readlines()])
    mesh_dict = set([f.strip().lower() for f in open(mesh_dict_file, "r", encoding="utf8").readlines()])
    stopwords_dict = set([f.strip().lower() for f in open(stopwords_english_dict_file, "r", encoding="utf8").readlines()])
    drugbank_dict = set([f.strip().lower() for f in open(drugbank_dict_file, "r", encoding="utf8").readlines()])
    icd_dict = set([f.strip().lower() for f in open(icd_dict_file, "r", encoding="utf8").readlines()])

    chv_map = [f.strip().split(",") for f in open(chv_dict_file, "r", encoding="utf8").readlines()]
    chv_map = [(a, float(b)) for a,b in chv_map]

    files = glob.iglob(path_to_data + "/*")
    #print("Processing %d files..." % (len(files)))

    shared.setConst(prefixes=prefixes)
    shared.setConst(sufixes=sufixes)
    shared.setConst(acronyms=acronyms)
    shared.setConst(outdir=outdir)
    shared.setConst(eng_dict=eng_dict)
    shared.setConst(mesh_dict=mesh_dict)
    shared.setConst(stopwords_dict=stopwords_dict)
    shared.setConst(drugbank_dict=drugbank_dict)
    shared.setConst(icd_dict=icd_dict)
    shared.setConst(chv_map=chv_map)

    shared.setConst(continuous=continuous)
    shared.setConst(preprocessing=preprocessing)

    print((sum(futures.map(process, files))))
    #print(sum(map(process, files)))

    print("Done!")

