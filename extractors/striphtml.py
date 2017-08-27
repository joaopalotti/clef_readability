import glob
from str2bool import str2bool
import sys, os
from auxiliar import get_content


path_to_documents = sys.argv[1]
out_path = sys.argv[2]
htmlremover = sys.argv[3]
forceperiod = str2bool(sys.argv[4])

for filename in glob.glob(os.path.join(path_to_documents, "*")):
    content = get_content(filename, htmlremover, forceperiod)

    outpath = os.path.join(out_path, os.path.basename(filename))

    print(filename)
    with open(outpath, "w", errors="surrogateescape") as fout:
        fout.write(content)


