import sys
from io import open
from collections import defaultdict, Counter
from heap import heap
import regex as re
from nltk import pos_tag
from nltk.stem.porter import PorterStemmer
from nltk.tag.mapping import map_tag


def weight(vertex, neighbors, C):
    """Return the sum of edge weights of the vertex.

    The edges under consideration are restricted to those endpoints
    which are in set C.

    """
    return sum(count for n, count in neighbors[vertex].items() if n in C)


def N(neighbors, C):
    """Neighbors restricted to set C."""
    return C.intersection(neighbors.keys())


def p_cores(vertices, neighbors):
    """p-cores algorithm from Batagelj & Zaveršnik

    Calculates the weighted k-cores decomposition. The weight function
    p() is hard-coded to the edge weights in the graph.

    References
    ==========
    Batagelj, Vladimir, and Matjaž Zaveršnik. "An O(m) algorithm for
    cores decomposition of networks." arXiv preprint cs/0310049
    (2003).

    """
    # XXX: either this is buggy, or the numbers in the paper
    # rousseau-ecir2015 are incorrect. Check this against networkx'
    # k-cores implementation?
    cores = {}

    C = set(v for v in vertices)
    p = {v: weight(v, neighbors, C) for v in vertices}

    min_heap = heap.heap([])

    for v in vertices:
        min_heap[v] = weight(v, neighbors, C)

    while len(C) > 0:
        top = min_heap.pop()
        C.remove(top)

        cores[top] = p[top]
        for v in N(neighbors[top], C):
            p[v] = max(p[top], weight(v, neighbors, C))
            min_heap[v] = p[v]

    return cores


def ngrams(lst, n=4):
    N = len(lst)
    if N < n:
        return
        yield

    for i in range(N-n):
        yield lst[i:i+n]


def add_edges(graph, lst):
    # N = len(lst)
    for i, a in enumerate(lst):
        for j, b in enumerate(lst):
            if i == j:
                continue
            graph[a][b] = 1
            # graph[a][b] += 1.0/(1+(i-j)**2)
            # graph[a][b] += 2.0**-(abs(i-j)-1)


def get_tokens(fname, stopwords):
    with open(fname, encoding='utf-8') as f:
        text = f.read()

    # text = re.sub(r'\d', '9', text)
    # word_re = re.compile(r'(\p{L}[\p{L}_-]+|\p{P}+)')
    word_re = re.compile(r'(\p{L}[\p{L}_-]+)')
    tokens = word_re.findall(text)

    # retain = set(['NOUN', 'ADJ', 'ADV', 'PROPN'])
    retain = set(['NOUN', 'ADJ'])
    pos_tagged = pos_tag(tokens)
    print "pos_tags:", " ".join("{}_{}".format(*t) for t in pos_tagged)
    tokens = [tok for tok, tag in pos_tag(tokens)
              if map_tag('en-ptb', 'universal', tag) in retain]

    tokens = [tok.lower() for tok in tokens]
    tokens = [tok for tok in tokens if tok not in stopwords]

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(tok) for tok in tokens]

    print "====="
    print "tokens:", " ".join(tokens)
    print "====="

    return tokens


if __name__ == '__main__':
    fname = sys.argv[1]

    with open('english_stopwords.txt', encoding='utf-8') as f:
        stopwords = set(f.read().split())
    toks = get_tokens(fname, stopwords)
    graph = defaultdict(Counter)

    for ngram in ngrams(toks, n=4):
        # print " ".join(ngram)
        add_edges(graph, ngram)

    print "graph({}):".format(len(graph))
    for a, counter in graph.items():
        for b, count in counter.items():
            print "{}\t{}\t{}".format(a, b, count)
    print "====="

    cores = p_cores(graph.keys(), graph)

    for stem, k_core in Counter(cores).most_common(50):
        print u"{}\t{}".format(stem, k_core).encode('utf-8')
