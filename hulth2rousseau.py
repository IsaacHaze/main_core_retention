import sys
from io import open
from collections import OrderedDict


from nltk.stem.porter import PorterStemmer


with open(sys.argv[1], encoding='utf-8') as f:
    data = f.read()

porter = PorterStemmer()

res = OrderedDict()
for keyphrase in data.split(';'):
    phrase = []
    for tok in keyphrase.split():
        tok = tok.lower()
        stem = porter.stem(tok)
        phrase.append(stem)
    phrase = " ".join(phrase)
    res[phrase] = None

print "\n".join(res.keys()).encode('utf-8')
            
