from byeHTML import byeHTML
import kenlm
import glob
import gzip
from features import find_encoding

filenames = glob.glob("../data/pool_2016/*")

model0 = kenlm.LanguageModel('../data/pool_2015/read0.justext.bin')
model1 = kenlm.LanguageModel('../data/pool_2015/read1.justext.bin')
model2 = kenlm.LanguageModel('../data/pool_2015/read2.justext.bin')
model3 = kenlm.LanguageModel('../data/pool_2015/read3.justext.bin')

for filename in filenames:
    encoding = find_encoding(filename)

    if filename.endswith(".gz"):
        with gzip.open(filename, mode="rt", encoding=encoding, errors="surrogateescape") as f:
            content = str(f.read()) # Explicitly convert from bytes to str
    else:
        with open(filename, encoding=encoding, errors="surrogateescape", mode="r") as f:
            content = f.read()

    text = byeHTML(content, preprocesshtml="justext", forcePeriod=False).get_text()

    print("%s,%.3f,%.3f,%.3f,%.3f" % (filename, model0.score(text), model1.score(text), model2.score(text), model3.score(text)))



