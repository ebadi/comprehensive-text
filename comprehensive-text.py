import xml.etree.ElementTree as ETree
import numpy as np
import sqlite3
import random
# python3 -m spacy download sv_core_news_lg
import spacy
from sklearn.cluster import AgglomerativeClustering


def distance(w1, w2):
    tokens = model(w1 + ' ' + w2)
    token1, token2 = tokens[0], tokens[1]
    sim = token1.similarity(token2)
    return sim


NUM_CLUSTER = 100
xmldata = "kelly.xml"
model = spacy.load('sv_core_news_lg')
prstree = ETree.parse(xmldata)
root = prstree.getroot()
print("Model and word list are loaded")


word_list = []
for element in root.iter('FormRepresentation'):
    values = []
    for f in list(element):
        values.append(f.attrib['val'])
    if len(values) != 9:  # HACK for elements with no gram attributes
        values.append('NA')
    word_list.append(values[0])

print(word_list)
print("Word list is created")

# Convert objects into numerical vectors based on Jaccard similarity
n_objects = len(word_list)
dist_matrix = np.zeros((n_objects, n_objects))
for i in range(n_objects):
    for j in range(i + 1, n_objects):
        dist_matrix[i, j] = dist_matrix[j, i] = distance(
            word_list[i], word_list[j])

# Perform Agglomerative Clustering
print(dist_matrix)
print("Distance matrix is created")
agglomerative_clustering = AgglomerativeClustering(n_clusters=NUM_CLUSTER)
labels = agglomerative_clustering.fit_predict(dist_matrix)

print(labels)
print("Each word is now labeled with its cluster id")

clusters_array = [[] for _ in range(NUM_CLUSTER)]

for l in range(len(labels)):
    clusters_array[labels[l]].append(word_list[l])

print(clusters_array)
print("Words are now clustered")
