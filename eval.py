import sys
import os
from glob import iglob


def load_unigrams(fname):
    unigrams = set([])
    with open(fname) as f:
        for line in f:
            for tok in line.split():
                unigrams.add(tok)
    return unigrams


def avg(lst):
    return sum(lst) / len(lst)


gold, eval = sys.argv[1], sys.argv[2]

gold_unigrams = {}
for fname in iglob(os.path.join(gold, '*')):
    bn, _  = os.path.basename(fname).split('.', 1)
    gold_unigrams[bn] = load_unigrams(fname)

eval_unigrams = {}
for fname in iglob(os.path.join(eval, '*')):
    bn, _  = os.path.basename(fname).split('.', 1)
    eval_unigrams[bn] = load_unigrams(fname)

ps = []
rs = []
f1s = []
for bn in gold_unigrams:
    try:
        g = gold_unigrams[bn]
        e = eval_unigrams[bn]
        i = g.intersection(e)
        p = float(len(i))/len(e)
        r = float(len(i))/len(g)
        try:
            f1 = 2*p*r/(p+r)
        except ZeroDivisionError:
            f1 = 0.0
    except KeyError:
        print >>sys.stderr, "ugh: {}".format(bn)
        p = 0.0
        r = 0.0
        f1 = 0.0

    print >>sys.stderr, "bn: {: >4}, p: {:1.3f}, r: {:1.3f}, f1: {:1.3f}".format(bn, p, r, f1)
    ps.append(p)
    rs.append(r)
    f1s.append(f1)

print "num:", len(f1s)
print "precision:",avg(ps)
print "recall:",avg(rs)
print "f1:",avg(f1s)
