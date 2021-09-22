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
import faiss

# Parameters

"""
Trains a model using Damian's idea of validation : validate maximising
the area under the Top-K curve

Trains faster and yields a better Top-K curve than main.py, but lower AUC
"""

input = sys.argv[1]
disable_training = False
disable_testing = False
enable_evaluation = False
min_no_vertices = 5
model = f'damian.h5'
mini_batch_size = 128
learning_rate = 0.0005
no_training_pairs = 1
no_validation_pairs = 1
no_testing_pairs = 10
validation_batch = 50_000

total_time = timeit.default_timer()
print("Computing total time")
print("timing,Time,Validation Loss")

# Implementation
#strategy = tf.distribute.MirroredStrategy()
test = tf.constant([1,-1])
with tf.device("/GPU:0"):#strategy.scope():#if __name__ == "__main__":
    from helpers import *
    from patchy_san import *
    from cnn import *

    with open(input) as file:
        data = json.load(file)

    train_map = get_PS_embeddings(data["train"], parameters_PS)
    valid_map = get_PS_embeddings(data["valid"], parameters_PS)
    testi_map = get_PS_embeddings(data["testi"], parameters_PS)
    data = None
    
        
    def separate(data_map):
    # Separate info on function (Name and number of similarities) and embeddings which will 
    # be grouped in a single tensor to computes the embeddings faster
        i = 0
        names = []
        PS_embeddings = []
        for fun, PS_es in data_map.items():
            for PS_e in PS_es:
                names.append((fun, len(PS_es), i))
                PS_embeddings.append(PS_e)
            i += len(PS_es)
        return names, PS_embeddings
    
    valid_names, valid_embeddings = separate(valid_map)
    batched_valid_embeddings = tf.data.Dataset.from_tensor_slices(valid_embeddings).batch(validation_batch)

    # Initialize graph embedding model
    #GE = CNN()
    if os.path.isfile(model):
        GE.load_weights(model)

    #GE.save_weights(model)
  
# Training / Validating
################################################################################################
    if not disable_training:
        # Evaluation - setup
        if enable_evaluation:
            f = os.path.join(filepath, os.path.splitext(new_filename)[0] + '.stats.csv')
            exists = True if os.path.isfile(f) else False
            csv_file = open(f, 'a')
            csv_writer = csv.DictWriter(csv_file, delimiter=',', quotechar='"',
                                        fieldnames=[
                                            'EPOCH',
                                            'LOSS',
                                            'AUC',
                                            'MEMORY[MB]',
                                            'TRAIN_DATA_TIME[s]',
                                            'TRAIN_TIME[s]',
                                            'VALID_DATA_TIME[s]',
                                            'VALID_TIME[s]'
                                        ])
            if not exists:
                csv_writer.writeheader()
        
        # Epochs
        max_roc_auc = 0
        not_improving_decount = 25
        epoch = 0
        while not_improving_decount > 0:#for epoch in range(epochs):
            not_improving_decount -= 1
            epoch += 1
            print('[>] Epoch {}:'.format(epoch))
  
            # Training
            print('\t[>] Training:')
  
            # Generate epoch's training data
            print('\t\ta. Generate training data:')
            train_data_time0 = timeit.default_timer()
            Ys, E1s, E2s = generate_CNN_input(train_map, no_training_pairs)
            if len(Ys) <= 0:
                print('\t[!] Terminating due to empty training data')
                exit(0)
                
            print('\t\tb. Zip training data:')
            dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(Ys), tf.data.Dataset.from_tensor_slices(E1s), tf.data.Dataset.from_tensor_slices(E2s)))
            batches = math.ceil(len(Ys)/mini_batch_size)
            Ys, E1s, E2s = None, None, None
  
            # Shuffle epoch's training data
            print('\t\tc. Shuffle training data')
            dataset = dataset.shuffle(len(dataset),reshuffle_each_iteration=False)
  
            # Generate mini batches
            print('\t\td. Generate mini batches')
            dataset = dataset.batch(mini_batch_size)
            
            #print('\t\te. Distribute training data:')
            #dataset = strategy.experimental_distribute_dataset(dataset)
            train_data_time1 = timeit.default_timer()
  
            # Train on mini batches
            print('\t\tf. Train on mini batches:')
            train_time0 = timeit.default_timer()
            train_fast(dataset, batches, learning_rate)
            train_time1 = timeit.default_timer()
            dataset = None
  
            # Validating
            print('\t[>] Validating:')
  
            # Generate epoch's validation data
            print('\t\ta. Generate validation data:')
            embeddings = tf.concat([GE(batch) for batch in batched_valid_embeddings], axis=0)
            embeddings = tf.math.l2_normalize(embeddings, axis=1).numpy()
            index = faiss.IndexFlatL2(25)
            index.add(embeddings)
            NUMBER_OF_NEIGHBORS = 200
            _, sorted_neighbors = index.search(embeddings, NUMBER_OF_NEIGHBORS)
            del embeddings, index
            
            score = 0.
            for node, (fun, similar, index) in enumerate(valid_names):
                raw_score = 0
                for K, neighbor in enumerate(sorted_neighbors[node,:NUMBER_OF_NEIGHBORS]):
                    raw_score += NUMBER_OF_NEIGHBORS - K if index <= neighbor and neighbor < index + similar else 0
                score += raw_score / similar
            score /= len(valid_names)
            roc_auc = score
  
            # Store model with best ROC AUC score
            if roc_auc > max_roc_auc:
                print('\t\td. Store model (best ROC AUC score)')
                max_roc_auc = roc_auc
                not_improving_decount = 25
                GE.save_weights(model)
            
            print(f"timing,{timeit.default_timer()-total_time}s,{roc_auc}", flush=True)
  
        print('[>] Model saved to file "{}"'.format(model))
        print('[>] Best ROC AUC score: {:.5f}'.format(max_roc_auc))
  
  # Testing
  ################################################################################################
    if not disable_testing:
        print('[>] Testing:')
        GE.load_weights(model)
      
        # Generate testing data
        print('\ta. Generate testing data:')
        Ys, E1s, E2s = generate_CNN_input(testi_map, no_testing_pairs)
        dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(E1s), tf.data.Dataset.from_tensor_slices(E2s)))
      
        # Calculate predictions and loss
        print('\tb. Calculate predictions and loss:')
        E1s, E2s = None, None
        dataset = dataset.batch(validation_batch)
        valid_time0 = timeit.default_timer()
        cosine_sim = tf.concat([cosine_similarities(GE(E1s), GE(E2s)) for E1s,E2s in dataset], axis=0)
        dataset = None
        difference = cosine_sim - tf.reshape(tf.dtypes.cast(Ys, tf.float32), [-1])
        squared_loss = difference * difference
        loss = tf.reduce_sum(squared_loss) / len(Ys)
        _Ys = 1.0 - (tf.math.acos(cosine_sim) / np.pi)
        print('\t\tLoss: {:.5f}'.format(loss))
      
        # Calculate ROC curve and ROC AUC score
        print('\tc. Calculate ROC curve and ROC AUC score:')
        fpr, tpr, _ = roc_curve(Ys, _Ys)
        roc_auc = auc(fpr, tpr)
        print('\t\tAUC:  {:.5f}'.format(roc_auc))
        
        with open(f"roc_curve_{input}.csv","w") as buffer:
            buffer.write("fpr,tpr\n")
            for fp,tp in zip(fpr,tpr):
                buffer.write(f"{fp},{tp}\n")
      
        print('[>] Model saved to file "{}"'.format(model))
        print('[>]  AUC score: {:.5f}'.format(roc_auc))

print("Total time : ",timeit.default_timer()-total_time)
