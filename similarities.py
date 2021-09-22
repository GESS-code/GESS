import json
import numpy as np
from random import sample
from progress.bar import IncrementalBar
import timeit
import sys

from sklearn.neighbors import BallTree, KDTree

"""
Computes the values of the Top-K curve
Returns precision and recall in CSV to the standard output

Uses files from compute_embeddings.py
Once the model is computed, we need to compute the function
embeddings before this curve.
"""

MAX_K = 200

LSH_NUMBER_OF_PROJECTIONS = 200
LSH_PROJECTION_SIZE = 9
LSH_PARAMETERS = LSH_NUMBER_OF_PROJECTIONS * LSH_PROJECTION_SIZE
LSH_W = 1200

# Files returned by compute_embeddings
NAMES = "names_complete_dataset.json.txt"
EMBEDDINGS = "embeddings_complete_dataset.json.npy"

with open(NAMES) as json_file:
    names = json.load(json_file)
embeddings = np.load(EMBEDDINGS)

print("Data loaded successfully")

computation_time = timeit.default_timer()
tree = KDTree(embeddings)
sorted_neighbors = tree.query(embeddings, k=200, return_distance=False)
print(f"200 neighbors found in {timeit.default_timer()-computation_time}s")

print("Computed neighbors")

recalls = [0 for _ in range(MAX_K)]
precisions = [0 for _ in range(MAX_K)]

with IncrementalBar(' ', max=len(names), suffix='%(percent)d%%') as bar:
    for node, (name, number_of_similar) in enumerate(names):
        current_count = 0
        for K in range(MAX_K):
            neighbor = sorted_neighbors[node,K]
            neighbor_name, _ = names[neighbor]
            if neighbor_name == name:
                current_count += 1
            precisions[K] += current_count / (K+1)
            recalls[K] += current_count / number_of_similar
        bar.next()

precisions = [precision / len(names) for precision in precisions]
recalls = [recall / len(names) for recall in recalls]
    
print("K, precision, recall")
for K in range(MAX_K):
    print(f"{K+1}, {precisions[K]}, {recalls[K]}")
