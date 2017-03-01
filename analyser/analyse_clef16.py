import sys
import pandas as pd
import numpy as np
from auxiliar import applyUnderstandabilityMap, normalizeFeatures, removeEmpties, clip_readability, removeDuplicates
from itertools import permutations
from sklearn import metrics, model_selection, linear_model, tree, dummy, svm, preprocessing, decomposition, ensemble
from sklearn import feature_selection

features_file = sys.argv[1]

qrels = pd.read_csv("../qrels/clef16/doc_all.mean")
doc_features = pd.read_csv(features_file)

# preprocess datasets:
applyUnderstandabilityMap(qrels)
doc_features = normalizeFeatures(doc_features)
doc_features = clip_readability(doc_features)

merged = pd.merge(qrels, doc_features)

relevant = merged[merged["rel"] > 0]
relevant = removeEmpties(relevant)

# Remove duplicated cols
relevant = removeDuplicates(relevant)

"""
labels = relevant["classunders"].sort_values()
for metric in [ u'rel', u'trust', u'unders',
       u'number_chars', u'number_words', u'number_sentences',
       u'number_syllables', u'number_polysyllable_words', u'difficult_words',
       u'longer_4', u'longer_6', u'longer_10', u'longer_13',
       u'flesch_reading_ease', u'flesch_kincaid_grade_level',
       u'coleman_liau_index', u'gunning_fog_index', u'smog_index',
       u'ari_index', u'lix_index', u'dale_chall_score', u'prefixes_found',
       u'sufixes_found', u'acronyms_found', u'numbers_found', u'eng_found',
       u'mesh_found', u'stopwords_found', u'drugbank_found', u'icd_found',
       u'chv_num', u'chv_mean', u'chv_sum']:
    pred = relevant[[metric, "classunders"]].sort_values(metric)["classunders"]
    print "%s: %.2f, %.2f" % (metric, metrics.precision_score(labels, pred, average="weighted"), metrics.accuracy_score(labels, pred))
    print metrics.confusion_matrix(labels, pred)

"""

remove_keys = ['filename', 'rel', 'trust', 'unders', 'classunders']
valid_features = list(set(relevant.keys()) - set(remove_keys))

def find_nan(X):
    problem = []
    for k in list(X.keys()):
        if X[X[k].apply(np.isnan)].shape[0] > 0:
            problem.append(k)
    return problem

X = relevant[valid_features]
Xnona = X.fillna(0.0)

sel = feature_selection.VarianceThreshold(threshold=0.8)
#Xnona = sel.fit_transform(Xnona)

#Xnona = preprocessing.normalize(Xnona, norm='l2')
#Xnona = decomposition.PCA(n_components=50).fit_transform(X)

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

gkg = {}
for k in set(relevant.keys()) - set(['filename', u'rel', u'trust', u'unders']):
    score = goodman_kruskal_gamma(relevant["classunders"].values, relevant[k].values)
    print("%s: %.4f" % (k,score))
    gkg[k] = score

acc = metrics.make_scorer(metrics.accuracy_score)



#model = linear_model.LinearRegression()
model = linear_model.LogisticRegression() # ~0.53
#model = linear_model.LogisticRegression(n_jobs=1, solver="liblinear", max_iter=1000, class_weight="balanced", C=1.0) # 0.53
#model = tree.ExtraTreeClassifier(min_samples_split=2, criterion="gini", class_weight="balanced", random_state=29) # 0.48
#model = tree.ExtraTreeClassifier(random_state=29)
#model = ensemble.RandomForestClassifier(random_state=29)
#model = svm.LinearSVC()
#model = dummy.DummyClassifier(strategy="most_frequent")
#model = dummy.DummyClassifier(strategy="stratified")
#print model
print(( "Mean Accuracy score: %.4f " % \
        (np.mean(model_selection.cross_val_score(model, Xnona, relevant["classunders"], cv=10, scoring=acc)))
     ))


def feature_importance(forest, names):
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, names[indices[f]], importances[indices[f]]))



    import matplotlib.pyplot as plt
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(list(range(X.shape[1])), importances[indices],
	   color="r", yerr=std[indices], align="center")
    plt.xticks(list(range(X.shape[1])), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()


# Many features are repeated....need to get rid of them.

def show_most_informative_features(names, clf, n=20):
    coefs_with_fns = sorted(zip(clf.coef_[0], names))
    top = list(zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1]))
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print(("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)))
