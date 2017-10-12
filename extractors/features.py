import sys
import gzip
import re
import os
import glob
import csv
from auxiliar import get_content
from readcalc import readcalc
from scoop import futures, shared
import numpy as np
import chardet

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

def process(filename):

    forcePeriod = shared.getConst('forcePeriod')
    preprocessing = shared.getConst('preprocessing')

    outdir = shared.getConst('outdir')

    prefixes = shared.getConst('prefixes')
    sufixes = shared.getConst('sufixes')
    acronyms = shared.getConst('acronyms')
    eng_dict = shared.getConst('eng_dict')
    med_dict = shared.getConst('med_dict')
    eng_med_dict = shared.getConst('eng_med_dict')
    mesh_dict = shared.getConst('mesh_dict')
    stopwords_dict = shared.getConst('stopwords_dict')
    drugbank_dict = shared.getConst('drugbank_dict')
    icd_dict = shared.getConst('icd_dict')
    chv_map = shared.getConst('chv_map')

    outfilename = outdir + "/" + filename.split("/")[-1]

    csv_file = open(outfilename, 'w')
    csv_writer = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)

    content = get_content(filename, htmlremover=preprocessing, forcePeriod=forcePeriod)

    # I am using byeHtml from auxiliar instead of readcalc
    calc = readcalc.ReadCalc(content, preprocesshtml=None)

    #calc = readcalc.ReadCalc(content, preprocesshtml=preprocessing, forcePeriod=True)
    #print("#Words after preprocessing (TRUE): %d" % (len(calc.get_words())))
    #print("#Sente after preprocessing (TRUE): %d" % (len(calc.get_sentences())))
    #calc = readcalc.ReadCalc(content, preprocesshtml=preprocessing, forcePeriod=False)
    #print("#Words after preprocessing (FALSE): %d" % (len(calc.get_words())))
    #print("#Sente after preprocessing (FALSE): %d" % (len(calc.get_sentences())))

    prefixes_found = count_prefixes(calc.get_words(), prefixes)
    sufixes_found = count_sufixes(calc.get_words(), sufixes)
    acronyms_found = count_acronyms(calc.get_words(), acronyms)
    numbers_found = count_numbers(calc.get_words())
    eng_found = count_dict_words(calc.get_words(), eng_dict)
    med_found = count_dict_words(calc.get_words(), med_dict)
    eng_med_found = count_dict_words(calc.get_words(), eng_med_dict)
    mesh_found = count_dict_words(calc.get_words(), mesh_dict, 3)
    stopwords_found = count_dict_words(calc.get_words(), stopwords_dict)
    drugbank_found = count_dict_words(calc.get_words(), drugbank_dict, 3)
    icd_found = count_dict_words(calc.get_words(), icd_dict, 3)
    chv_num, chv_mean, chv_sum = calc_chv(calc.get_words(), chv_map)

    filename = os.path.basename(filename)


    print("Found %d words." % len(calc.get_words()))
    print("Found %d sentences." % len(calc.get_sentences()))
    readability_row = [filename] + list(calc.get_all_metrics())
    readability_row.extend( [prefixes_found, sufixes_found, acronyms_found, numbers_found, eng_found, med_found, \
            eng_med_found, mesh_found, stopwords_found, drugbank_found, icd_found, chv_num, chv_mean, chv_sum] )

    # Header: filename, number_chars, number_words, number_types, number_sentences, number_syllables, number_polysyllable_words, difficult_words, longer_4, longer_6, longer_10, longer_13, flesch_reading_ease, flesch_kincaid_grade_level, coleman_liau_index, gunning_fog_index, smog_index, ari_index, lix_index, dale_chall_score, prefixes_found, sufixes_found, acronyms_found, numbers_found, eng_found, med_found, eng_med_found, mesh_found, stopwords_found, drugbank_found, icd_found, chv_num, chv_mean, chv_sum

    csv_writer.writerow(readability_row)
    csv_file.flush()
    csv_file.close()
    print(("File %s saved." % (outfilename)))
    return 1

## ========================================= ##
## ---------------Main---------------------- ##
## ========================================= ##

if __name__ == "__main__":

    if len(sys.argv) <= 2:
        script_name = os.path.basename(__file__)
        print(("USAGE: python %s <PATH_TO_DATA> <OUT_DIR>" % (script_name)))
        sys.exit(0)

    forcePeriod=False            # Options: True, False
    preprocessing = None         # Options: justext, bs4, boi, None

    print(("PARAMETERS: ", sys.argv))
    path_to_data = sys.argv[1]
    outdir = sys.argv[2]
    print(("ForcePeriod: %s, Preprocessing: %s" % (forcePeriod, preprocessing)))

    sufix_file = os.path.join(resource_path, "suffixes")
    prefix_file = os.path.join(resource_path, "prefixes")
    acronyms_file = os.path.join(resource_path, "acronyms.txt")

    stopwords_english_dict_file = os.path.join(resource_path, "stopwords.txt")
    general_english_dict_file = os.path.join(resource_path, "en-merged.dic")
    openmedical_english_dict_file = os.path.join(resource_path, "en_US_OpenMedSpel_1.0.0.dic")
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
    med_dict = set([f.strip().lower() for f in open(openmedical_english_dict_file, "r", encoding="ISO-8859-2").readlines()])
    eng_med_dict = eng_dict.union(med_dict)
    mesh_dict = set([f.strip().lower() for f in open(mesh_dict_file, "r", encoding="utf8").readlines()])
    stopwords_dict = set([f.strip().lower() for f in open(stopwords_english_dict_file, "r", encoding="utf8").readlines()])
    drugbank_dict = set([f.strip().lower() for f in open(drugbank_dict_file, "r", encoding="utf8").readlines()])
    icd_dict = set([f.strip().lower() for f in open(icd_dict_file, "r", encoding="utf8").readlines()])

    chv_map = [f.strip().split(",") for f in open(chv_dict_file, "r", encoding="utf8").readlines()]
    chv_map = [(a, float(b)) for a,b in chv_map]

    files = glob.glob(path_to_data + "/*")
    # Debug only:
    # print("Processing %d files..." % (len(list(files))))

    shared.setConst(prefixes=prefixes)
    shared.setConst(sufixes=sufixes)
    shared.setConst(acronyms=acronyms)
    shared.setConst(outdir=outdir)
    shared.setConst(eng_dict=eng_dict)
    shared.setConst(med_dict=med_dict)
    shared.setConst(eng_med_dict=eng_med_dict)
    shared.setConst(mesh_dict=mesh_dict)
    shared.setConst(stopwords_dict=stopwords_dict)
    shared.setConst(drugbank_dict=drugbank_dict)
    shared.setConst(icd_dict=icd_dict)
    shared.setConst(chv_map=chv_map)

    shared.setConst(forcePeriod=forcePeriod)
    shared.setConst(preprocessing=preprocessing)

    total = (sum(futures.map(process, files)))
    #total = (sum(map(process, files)))

    print()
    print("*" * 30)
    print(("Done! Processed %d documents. " % (total)))
    print("*" * 30)
    print()

