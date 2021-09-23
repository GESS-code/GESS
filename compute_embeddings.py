import os
import numpy as np
import tensorflow as tf
import csv
from datetime import datetime
import math
import memory_profiler
from progress.bar import IncrementalBar
import random
import re
from shutil import copyfile
from sklearn.metrics import roc_curve, auc
import timeit
import json
import sys

"""
Computes the embeddings of the functions in a dataset.
"""

## Parameters

input = sys.argv[1]     # Dataset
model = "MODEL"         # Trained model

with tf.device("/GPU:0"):#strategy.scope():#if __name__ == "__main__":
    from helpers import *
    from patchy_san import *
    from cnn import *

    with open(input) as file:
        data = json.load(file)

    testi_map = get_PS_embeddings(data["testi"], parameters_PS)
    del data

    # Initialize graph embedding model
    assert os.path.isfile(model)
    GE.load_weights(model)

    # Separate info on function (Name and number of similarities) and embeddings which will 
    # be grouped in a single tensor to computes the embeddings faster
    names, embeddings = zip(*[((fun, len(PS_es)), PS_e) for fun, PS_es in testi_map.items() for PS_e in PS_es])
    del testi_map
    
    embeddings = tf.stack(embeddings)
    
    def do(batch):
        print(tf.shape(batch))
        return GE(batch)
    computation_time = timeit.default_timer()
    embeddings = do(embeddings)
    
    embeddings = tf.math.l2_normalize(embeddings, axis=1)
    print(f"{len(names)} embeddings computed and normalized in {timeit.default_timer()-computation_time}s")
    
    np.save(f"embeddings_{input}.npy", embeddings.numpy())
    with open(f"names_{input}.txt", "w") as file:
        file.write(json.dumps(names, indent=4))

print("Done\n")
