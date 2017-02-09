
def clip_readability(df):

    school_year_metrics = ["ari_index", "smog_index", "gunning_fog_index", "flesch_kincaid_grade_level"]

    # Max ARI Index is 15:
    for k in df.keys():
        for metric_name in school_year_metrics:
            if metric_name in k:
                # We clip in 20 years in school ~ College degree.
                df[k] = df[k].clip(0.0, 20.0)

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
            df[k] = df[k].clip(0.0, 20.0)

    return df


def mapunders(understandability):
    if understandability <= 70:
        return 0
    else:
        return 1
    """
    elif understandability <= 80:
        return 1
    else:
        return 2
    """
def applyUnderstandabilityMap(qrels):
    qrels["classunders"] = qrels["unders"].apply(mapunders)

def normalizeFeatures(doc_features):

    keys = doc_features.keys()

    fields = ['prefixes_found', 'sufixes_found', 'acronyms_found', 'numbers_found', 'eng_found',
                'mesh_found', 'stopwords_found', 'drugbank_found', 'icd_found',
                'longer_4', 'longer_6', 'longer_10', 'longer_13',
                'chv_num', 'chv_sum',
                'difficult_words',
                ]

    for k in keys:
        for f in fields:
            if f in k:
                #print("Found %s in %s" % (f, k))
                suffix = k.rsplit(f,1)[1]
                #print("Suffix: %s" % (suffix))
                #
                #doc_features[k + "_normalized"] = doc_features[k] / doc_features["number_words" + suffix]
                doc_features[k] = doc_features[k] / doc_features["number_words" + suffix]
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

    endings = {"_jsp": ["_jst_fp", "_jst_nfp"], "_bs4": ["_bs4_fp","_bs4_nfp"]}

    for field in duplicates:
        for ending in endings.keys():
            for subending in endings[ending]:
                df[field + ending] = df[field + subending]

    for field in duplicates:
        for group in endings.keys():
            for ending in endings[group]:
                del df[field + ending]
    return df




