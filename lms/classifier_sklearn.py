import pandas as pd
import numpy as np
import sys
import glob
import os
from auxiliar import get_content
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes
from sklearn import decomposition
from sklearn import ensemble
from sklearn import neural_network
from sklearn import linear_model
from sklearn import svm

reddit_in = sys.argv[1]
pubmed_in = sys.argv[2]
enwikipedia_in = sys.argv[3]
path_to_files = sys.argv[4]
model = sys.argv[5]
outname = sys.argv[6]

task = sys.argv[7]
preprocess = sys.argv[8]
fp = sys.argv[9]

print("This will create %s" % (outname))

reddit = pd.read_csv(reddit_in)
pubmed = pd.read_csv(pubmed_in)
enwiki = pd.read_csv(enwikipedia_in)

# goodbye empties...
reddit = reddit[~reddit["text"].isnull()]
enwiki = enwiki[~enwiki["content"].isnull()]
pubmed = pubmed[~pubmed["abstract"].isnull()]

# decrease size of pubmed data
pubmed = pubmed.sample(40000)

y = np.concatenate((np.ones(reddit.shape[0]), np.ones(enwiki.shape[0]) + 1, np.ones(pubmed.shape[0]) + 2))
X = pd.concat((reddit["text"], enwiki["content"], pubmed["abstract"]))

#count_vect = CountVectorizer()
#X_counts = count_vect.fit_transform(X)

tfidf_transformer = TfidfVectorizer(sublinear_tf=True, use_idf=True)
X_tfidf = tfidf_transformer.fit_transform(X)

#for components in range(10, 501, 20):
components = 10

LSA = decomposition.TruncatedSVD(n_components=components, random_state=42, algorithm='arpack')
X_lsa = LSA.fit_transform(X_tfidf, y)

#clf = MultinomialNB().fit(Xlsa, y)
if model == "rf":
    clf = ensemble.RandomForestRegressor(random_state=42).fit(X_lsa, y) if task == "reg" else ensemble.RandomForestClassifier(random_state=42).fit(X_lsa, y)
elif model == "nn":
    clf = neural_network.MLPRegressor(random_state=42).fit(X_lsa, y) if task == "reg" else neural_network.MLPClassifier(random_state=42).fit(X_lsa, y)
elif model == "gbr":
    clf = ensemble.GradientBoostingRegressor(random_state=42).fit(X_lsa, y) if task == "reg" else ensemble.GradientBoostingClassifier(random_state=42).fit(X_lsa, y)
elif model == "lr":
    clf = linear_model.LinearRegression(n_jobs=-1).fit(X_lsa, y) if task == "reg" else linear_model.LogisticRegression().fit(X_lsa, y)
elif model == "svr":
    clf = svm.SVR().fit(X_lsa, y) if task == "reg" else svm.SVC().fit(X_lsa, y)
elif model == "nb":
    clf = naive_bayes.MultinomialNB().fit(X_tfidf, y)
    #clf = naive_bayes.MultinomialNB().fit(X_tfidf, y) if task == "reg" else naive_bayes.MultinomialNB().fit(X_lsa, y)

testdf = []
for d in glob.glob(os.path.join(path_to_files, "*")):
    content = get_content(d, htmlremover=None)
    testdf.append((os.path.basename(d), content))
testdf = pd.DataFrame(testdf, columns=["filename","content"])

X_new_tfidf = tfidf_transformer.transform(testdf["content"])
if model == "nb" and task == "reg":
    predictions = clf.predict(X_new_tfidf)
else:
    X_new_lsa = LSA.transform(X_new_tfidf)
    predictions = clf.predict(X_new_lsa)

testdf["%s_%s_%s_%s" % (model, task, preprocess, fp)] = predictions

# Save testdf
testdf[["filename","%s_%s_%s_%s" % (model, task, preprocess, fp)]].to_csv(outname, index=False)


"""
print( "Done with %s - created %s" % (model,outname) )
qrels_file = "../qrels/clef16/doc_understandability.mean"
qrels = pd.read_csv("../qrels/clef16/doc_understandability.mean")
merged = pd.merge(qrels, testdf)
print("Components %d -- %.4f" % (components, merged.corr()["rel"]["predictions"]))
"""
