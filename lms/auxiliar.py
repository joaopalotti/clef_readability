from nltk import tokenize
import gzip
import chardet
from byeHTML import byeHTML

def get_words(text):
    word_tokenizer = tokenize.TreebankWordTokenizer()
    words = [w.strip('!@#$%^&*()[];/,.\'').lower()
            for w in word_tokenizer.tokenize(text) if w.strip() and w.strip('!@#$%^&*()[];/,.\'').isalpha()]
    return words

def find_encoding(doc_full_path):
    # http://chardet.readthedocs.io/en/latest/usage.html
    # This method uses the traditional chardet to find out the encoding used in a file
    if doc_full_path.endswith(".gz"):
        f = gzip.open(doc_full_path, mode="rb")
    else:
        f = open(doc_full_path, mode="rb")

    rawdata = f.read()
    return chardet.detect(rawdata)["encoding"]


def get_content(filename, htmlremover="justext", forcePeriod=False):
    encoding = find_encoding(filename)

    if filename.endswith(".gz"):
        with gzip.open(filename, mode="rt", encoding=encoding, errors="surrogateescape") as f:
            content = str(f.read()) # Explicitly convert from bytes to str
    else:
        with open(filename, encoding=encoding, errors="surrogateescape", mode="r") as f:
            content = f.read()

    content = byeHTML(content, preprocesshtml=htmlremover, forcePeriod=forcePeriod).get_text()
    return content

