
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
    """
    if understandability <= 70:
        return 0
    else:
        return 1
    """
    if understandability < 33:
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
                'acronyms_found', 'prefixes_found', 'sufixes_found', 'numbers_found',
                'stopwords_found', 'eng_found', 'number_words', 'number_of_chars',
                'difficult_words', 'number_polysyllable_words', 'number_syllables',
                'longer_4', 'longer_6', 'longer_10', 'longer_13',
                # Medical Vocabularies:
                'drugbank_found', 'icd_found', 'mesh_found',
                'chv_num', 'chv_sum', 'chv_mean'
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
                print("Found %s in %s" % (f, k))
                suffix = k.rsplit(f,1)[1]
                prefix = f
                print("Prefix: %s Suffix: %s" % (prefix, suffix))

                # this is the case for the html metrics, as I did not use bs4 nor justext to preprocess the text
                if len(suffix) == 0:
                    # I am arbitrarily chosen the number of sentences and words from bs4, not force period
                    suffix = "_bs4_nfp"

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





