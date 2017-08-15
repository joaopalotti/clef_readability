import re
from itertools import permutations
import pandas as pd
from sklearn import feature_selection

def clip_readability(df, school_cap=25):

    school_year_metrics = ["ari_index", "smog_index", "gunning_fog_index", "flesch_kincaid_grade_level", "coleman_liau_index"]

    for k in df.keys():
        for metric_name in school_year_metrics:
            if metric_name in k:
                # We clip in 25 years in school ~ College degree.
                df[k] = df[k].clip(0.0, school_cap)

    # Max flesch reading ease [0--100]:
    for k in df.keys():
        if "flesch_reading_ease" in k:
            df[k] = df[k].clip(0.0, 100.0)


    # Max LIX: 0-100 (actually more than 60 is already very difficult text)
    # http://www.readabilityformulas.com/the-LIX-readability-formula.php
    for k in df.keys():
        if "lix_index" in k:
            df[k] = df[k].clip(0.0, 100.0)

    # Max Dale-chall-score [0-20]
    # http://www.readabilityformulas.com/new-dale-chall-readability-formula.php
    for k in df.keys():
        if "dale_chall_score" in k:
            df[k] = df[k].clip(0.0, school_cap)

    return df


def mapunders(understandability):
    """
    if understandability <= 70:
        return 0
    else:
        return 1
    """
    if understandability <= 33:
        return 0
    elif understandability <= 66:
        return 1
    else:
        return 2
def applyUnderstandabilityMap(qrels):
    qrels["classunders"] = qrels["unders"].apply(mapunders)

def normalizeFeatures(doc_features):

    keys = doc_features.keys()

    fields = [  # General metrics:
                'acronyms_found', 'numbers_found',
                'stopwords_found', 'eng_found', 'number_words', 'number_chars',
                'difficult_words', 'number_polysyllable_words', 'number_syllables',
                'longer_4', 'longer_6', 'longer_10', 'longer_13', 'number_sentences',
                # Medical Vocabularies:
                'drugbank_found', 'icd_found', 'mesh_found',
                'chv_num', 'chv_sum', 'chv_mean', 'sufixes_found', 'prefixes_found',
                # html metrics:
                'n_abbrs','n_as','n_blockquotes','n_bolds','n_cites','n_divs','n_dls','n_forms','n_h1s','n_h2s','n_h3s',
                'n_h4s','n_h5s','n_h6s','n_hs','n_imgs', 'n_inputs', 'n_links', 'n_lists','n_ols','n_ps','n_qs', 'n_scripts',
                'n_spans','n_table','n_uls',
                # MeSH and CHV counts:
                'n_msh_concepts', 'n_msh_dysn_concepts', 'n_sosy_concepts',
                'n_chv_concepts', 'n_chv_dsyn_concepts', 'n_sosy_chv_concepts',
                'avg_tree_length_all', 'avg_tree_length_dysn', 'avg_tree_length_sosy',
                'avg_chv_value_all', 'avg_chv_value_dysn', 'avg_chv_value_sosy',
                ]

    for k in keys:
        for f in fields:
            if f in k:
                #print("Found %s in %s" % (f, k))
                suffix = k.rsplit(f,1)[1]
                prefix = f
                #print("Prefix: %s Suffix: %s" % (prefix, suffix))

                # this is the case for the html metrics, as I did not use bs4 nor justext to preprocess the text
                if len(suffix) == 0:
                    # I am arbitrarily choosing the number of sentences and words from jst, not force period
                    suffix = "_jst_nfp"
                    print("Could not find suffix for %s" % (k))

                doc_features[prefix + "_per_word" + suffix] = doc_features[k] / doc_features["number_words" + suffix]
                doc_features[prefix + "_per_sentence" + suffix] = doc_features[k] / doc_features["number_sentences" + suffix]


    return doc_features

def removeEmpties(doc_features):

    print("Doc_features had %d rows" % (doc_features.shape[0]))
    keys = [k for k in doc_features.keys() if "number_words" in k]

    for k in keys:
        doc_features = doc_features[doc_features[k] > 0]

    print("Now Doc_features has %d rows" % (doc_features.shape[0]))
    return doc_features

def removeDuplicates(df):
    duplicates = ["number_chars", "number_words", "number_syllables", "number_polysyllable_words",
            "difficult_words", "longer_4", "longer_6", "longer_10", "longer_13", "prefixes_found",
            'sufixes_found', 'acronyms_found', 'numbers_found', 'eng_found', 'mesh_found',
            'stopwords_found', 'drugbank_found', 'icd_found', 'chv_num', 'chv_mean', 'chv_sum']

    endings = {"_jst": ["_jst_fp", "_jst_nfp"], "_bs4": ["_bs4_fp","_bs4_nfp"], "_boi": ["_boi_fp","_boi_nfp"]}

    for field in duplicates:
        for ending in endings.keys():
            for subending in endings[ending]:
                df[field + ending] = df[field + subending]

    for field in duplicates:
        for group in endings.keys():
            for ending in endings[group]:
                del df[field + ending]
    return df

def clean(s):
    s = re.sub('_boi', '', s)
    s = re.sub('_bs4', '', s)
    s = re.sub('_jst', '', s)
    s = re.sub('_nopreprocess', '', s)
    s = re.sub('_fp', '', s)
    s = re.sub('_nfp', '', s)
    s = re.sub('_per_word', '', s)
    s = re.sub('_per_sentence', '', s)
    return s

def clean_preprocess(s):
    s = re.sub('_bs4', '', s)
    s = re.sub('_boi', '', s)
    s = re.sub('_jst', '', s)
    return s

def create_filters(df, field):
    df["bs4"] = df[field].apply(lambda x: "bs4" in x)
    df["boi"] = df[field].apply(lambda x: "boi" in x)
    df["jst"] = df[field].apply(lambda x: "jst" in x)
    df["npro"] = df[field].apply(lambda x: "nopreprocess" in x)
    df["fp"] = df[field].apply(lambda x: "fp" in x)
    df["nfp"] = df[field].apply(lambda x: "nfp" in x)
    df["pw"] = df[field].apply(lambda x: "per_word" in x)
    df["ps"] = df[field].apply(lambda x: "per_sentence" in x)
    df["html"] = df[field].apply(lambda x: x.startswith("n_"))

    df["root"] = df[field].apply(lambda x: clean(x))
    df["root_no_preprocess"] = df[field].apply(lambda x: clean_preprocess(x))

def topFeatures(X, y, K, feature_names, score_func=feature_selection.chi2):
    fs = feature_selection.SelectKBest(score_func=score_func).fit(X,y)
    features = pd.DataFrame(zip(feature_names, fs.scores_), columns=["feature", "value"])
    return features[["feature","value"]].sort_values(by="value", ascending=False).head(K)

def goodman_kruskal_gamma(m, n):
    """
    compute the Goodman and Kruskal gamma rank correlation coefficient;
    this statistic ignores ties is unsuitable when the number of ties in the
    data is high. it's also slow.
    >> x = [2, 8, 5, 4, 2, 6, 1, 4, 5, 7, 4]
    >> y = [3, 9, 4, 3, 1, 7, 2, 5, 6, 8, 3]
    >> goodman_kruskal_gamma(x, y)
    0.9166666666666666
    from https://github.com/shilad/context-sensitive-sr/blob/master/SRSurvey/src/python/correlation.py
    """
    if len(m)==0  or len(n) == 0:
        print("ERROR!!")
        return 0.0

    if min(n) == max(n) or min(m) == max(m):
        print("No variance!!! ERROR!!")
        return 0.0

    num = 0
    den = 0
    for (i, j) in permutations(list(range(len(m))), 2):
        m_dir = m[i] - m[j]
        n_dir = n[i] - n[j]
        sign = m_dir * n_dir
        if sign > 0:
            num += 1
            den += 1
        elif sign < 0:
            num -= 1
            den += 1
    return num / float(den)

