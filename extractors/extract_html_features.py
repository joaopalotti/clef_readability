from bs4 import BeautifulSoup
from auxiliar import get_content
from multiprocessing import Pool
from glob import glob
import os
import sys
import pandas as pd

path_to_files = sys.argv[1]

# Runs with Python 3.5, 2.7

def extract_html_features(filename):
    #print("Processing %s..." % (filename))
    try:
        html_content = get_content(filename, htmlremover=None)
        soup = BeautifulSoup(html_content, "html.parser")
    except:
        return {"filename": os.path.basename(filename)}

    features = {}
    features["filename"] = os.path.basename(filename)
    features["n_as"] = len(soup.find_all('a'))
    features["n_ps"] = len(soup.find_all('p'))
    features["n_qs"] = len(soup.find_all('q'))
    features["n_blockquotes"] = len(soup.find_all('blockquote'))
    features["n_abbrs"] = len(soup.find_all('abbr'))
    features["n_cites"] = len(soup.find_all('cite'))
    features["n_bolds"] = len(soup.find_all('b'))
    features["n_ols"] = len(soup.find_all('ol'))
    features["n_uls"] = len(soup.find_all('ul'))
    features["n_dls"] = len(soup.find_all('dl'))
    features["n_lists"] = features["n_ols"] + features["n_uls"] +  features["n_dls"]
    features["n_h1s"] = len(soup.find_all('h1'))
    features["n_h2s"] = len(soup.find_all('h2'))
    features["n_h3s"] = len(soup.find_all('h3'))
    features["n_h4s"] = len(soup.find_all('h4'))
    features["n_h5s"] = len(soup.find_all('h5'))
    features["n_h6s"] = len(soup.find_all('h6'))
    features["n_hs"] = features["n_h1s"] + features["n_h2s"] + features["n_h3s"] + features["n_h4s"] + features["n_h5s"]
    features["n_imgs"] = len(soup.find_all('img'))
    features["n_table"] = len(soup.find_all('table'))
    features["n_divs"] = len(soup.find_all('div'))
    features["n_forms"] = len(soup.find_all('form'))
    features["n_spans"] = len(soup.find_all('span'))
    features["n_scripts"] = len(soup.find_all('script'))
    features["n_inputs"] = len(soup.find_all('input'))
    features["n_links"] = len(soup.find_all('link'))
    return features


p = Pool()
files = glob(os.path.join(path_to_files,"*"))
features = p.map(extract_html_features, files)
df = pd.DataFrame(features)

df.to_csv(sys.stdout, index=False)

