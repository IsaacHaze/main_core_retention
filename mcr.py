# -*- coding: utf-8 -*-
import sys
import os
from io import open
from collections import defaultdict, Counter
from heap import heap
import regex as re

from nltk import pos_tag
from nltk.stem.porter import PorterStemmer
from nltk.tag.mapping import map_tag
from nltk.tokenize.punkt import PunktSentenceTokenizer

import networkx as nx
from networkx.algorithms.core import k_core, core_number


def load_stopwords(language):
    if language != 'english':
        raise
    with open(os.path.join(os.path.dirname(__file__),
                           'english_stopwords.txt'), encoding='utf-8') as f:
        stopwords = set(f.read().split())
    return stopwords


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
    p() is hard-coded to be the edge weights in the graph.

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
            if not graph.has_edge(a, b):
                graph.add_edge(a, b, weight=0)
            graph[a][b]['weight'] += 1
            # graph[a][b] += 1.0/(1+(i-j)**2)
            # graph[a][b] += 2.0**-(abs(i-j)-1)


def get_tokens(fname, stopwords, filter_pos=True, filter_stopwords=True,
               stem_words=True):
    with open(fname, encoding='utf-8') as f:
        text = f.read()

    if split_paragraphs:
        texts = re.split(r'\n(?:\s*\n)+', text)
    else:
        texts = [text]

    tokens = []
    for text in texts:
        # text = re.sub(r'\d', '9', text)
        # word_re = re.compile(r'(\p{L}[\p{L}_-]+|\p{N}+)')
        word_re = re.compile(r'(\p{L}[\p{L}_-]+)')
        tokens = word_re.findall(text)

        if filter_pos:
            # retain = set(['NOUN', 'ADJ', 'ADV', 'PROPN'])
            retain = set(['NOUN', 'ADJ'])
            pos_tagged = pos_tag(tokens)
            # print "pos_tags:", " ".join("{}_{}".format(*t) for t in pos_tagged)
            tokens = [tok for tok, tag in pos_tagged
                      if map_tag('en-ptb', 'universal', tag) in retain]

        if filter_stop:
            tokens = [tok.lower() for tok in tokens]
            tokens = [tok for tok in tokens if tok not in stopwords]

        if stem_words:
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(tok) for tok in tokens]

        # print "====="
        # print "tokens:", " ".join(tokens)
        # print "====="

    return tokens


def main_core_retention(text, stopwords='english', split_paragraphs=False,
                        split_sentences=False, filter_pos=True,
                        filter_stopwords=True, stem_words=True):
    stopwords = load_stopwords(stopwords)

    if split_paragraphs:
        texts = re.split(r'\n(?:\s*\n)+', text)
    else:
        texts = [text]

    if split_sentences:
        p = PunktSentenceTokenizer()
        texts = [sent for sent in p.tokenize(text) for text in texts]

    phrases = []
    for text in texts:
        # text = re.sub(r'\d', '9', text)
        # word_re = re.compile(r'(\p{L}[\p{L}_-]+|\p{N}+)')
        word_re = re.compile(r'(\p{L}[\p{L}_-]+)')
        tokens = word_re.findall(text)

        if filter_pos:
            # retain = set(['NOUN', 'ADJ', 'ADV', 'PROPN'])
            retain = set(['NOUN', 'ADJ'])
            pos_tagged = pos_tag(tokens)
            # print "pos_tags:", " ".join("{}_{}".format(*t) for t in pos_tagged)
            tokens = [tok for tok, tag in pos_tagged
                      if map_tag('en-ptb', 'universal', tag) in retain]

        if filter_stopwords:
            tokens = [tok.lower() for tok in tokens]
            tokens = [tok for tok in tokens if tok not in stopwords]

        if stem_words:
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(tok) for tok in tokens]

        # print "====="
        # print "tokens:", " ".join(tokens)
        # print "====="
        phrases.append(tokens)

    graph = build_cooccur_graph(phrases)

    main_core = k_core(graph)
    # for sub_g in nx.connected_components(main_core):
    #     print sub_g
    #     print "---"
    # sys.exit(1)
    return main_core.nodes()


def build_cooccur_graph(phrases):
    graph = nx.Graph()

    for phrase in phrases:
        for ngram in ngrams(phrase, n=3):
            # print " ".join(ngram)
            add_edges(graph, ngram)

    graph.remove_edges_from(graph.selfloop_edges())
    return graph


def read_text(fname):
    with open(fname, encoding='utf-8') as f:
        text = f.read()
    return text


if __name__ == '__main__':
    fname = sys.argv[1]
    text = read_text(fname)

    main_core = main_core_retention(text, split_paragraphs=False, split_sentences=False, stem_words=False)


    print "main core ({}):".format(len(main_core))
    print "\n".join("- {}".format(term) for term in main_core)
    # print sorted(core_number(graph).items(), key=lambda t:t[1], reverse=True)
