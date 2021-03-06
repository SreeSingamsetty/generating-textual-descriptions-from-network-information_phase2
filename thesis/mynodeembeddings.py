# -*- coding: utf-8 -*-
from node2vec import Node2Vec
import networkx as nx
import gensim

# Generate walks
graph = nx.read_edgelist("27testing.csv",delimiter="\t", encoding="utf-8")
node2vec = Node2Vec(graph, dimensions=128, walk_length=80, num_walks=10)
model = node2vec.fit(window=10)

model.wv.save_word2vec_format("27testing.emb")
