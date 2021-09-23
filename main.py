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

## Parameters

# Dataset and model. If model exists, it will be used and retrained starting from
# where it was. Otherwise it will be created
input = sys.argv[1]
model = f'{input}-GESS.h5'

# Training and testing.
# Attention : Disabling training when the model doesn't exist will result in an error
disable_training = False
disable_testing = False

# ML Parameters
mini_batch_size = 128
learning_rate = 0.0005
no_training_pairs = 1
no_validation_pairs = 1
no_testing_pairs = 10

validation_batch = 50_000 # Because all data doesn't fit at once in GPU memory. Need to do it step by step

total_time = timeit.default_timer()
print("Computing total time")
print("timing,Time,Validation Loss")

# Implementation
with tf.device("/GPU:0"):
    from helpers import *
    from patchy_san import *
    from cnn import *

    # Load data
    with open(input) as file:
        data = json.load(file)

    train_map = get_PS_embeddings(data["train"], parameters_PS)
    valid_map = get_PS_embeddings(data["valid"], parameters_PS)
    testi_map = get_PS_embeddings(data["testi"], parameters_PS)
    del data

    # Initialize graph embedding model
    if os.path.isfile(model):
        GE.load_weights(model)

# Training / Validating
################################################################################################
    if not disable_training:
        
        # Epochs
        max_roc_auc = 0
        not_improving_count = 0
        epoch = 0
        # Continue if improvement is done in the last 25 epochs
        while not_improving_count < 25:
            not_improving_count += 1
            epoch += 1
            print(f'[>] Epoch {epoch}:')
  
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
            del Ys, E1s, E2s
  
            print('\t\tc. Shuffle training data')
            dataset = dataset.shuffle(len(dataset),reshuffle_each_iteration=False)
  
            print('\t\td. Generate mini batches')
            dataset = dataset.batch(mini_batch_size)
  
            # Train on mini batches
            print('\t\tf. Train on mini batches:')
            train_fast(dataset, batches, learning_rate)
            del dataset

            
            # Validating
            print('\t[>] Validating:')
  
            print('\t\ta. Generate validation data:')
            Ys, E1s, E2s = generate_CNN_input(valid_map, no_validation_pairs)
            dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(E1s), tf.data.Dataset.from_tensor_slices(E2s)))
            del E1s, E2s
            dataset = dataset.batch(validation_batch)
  
            print('\t\tb. Calculate predictions and loss:')
            cosine_sim = tf.concat([cosine_similarities(GE(E1s), GE(E2s)) for E1s,E2s in dataset], axis=0)
            del dataset
            difference = cosine_sim - tf.reshape(tf.dtypes.cast(Ys, tf.float32), [-1])
            squared_loss = difference * difference
            loss = tf.reduce_sum(squared_loss) / len(Ys)
            _Ys = 1.0 - (tf.math.acos(cosine_sim) / np.pi)
            print('\t\t\tLoss: {:.5f}'.format(loss))
  
            print('\t\tc. Calculate ROC curve and ROC AUC score:')
            fpr, tpr, _ = roc_curve(Ys, _Ys)
            roc_auc = auc(fpr, tpr)
            print('\t\t\tAUC:  {:.5f}'.format(roc_auc))
  
            # Store model with best ROC AUC score
            if roc_auc > max_roc_auc:
                print('\t\td. Store model (best ROC AUC score)')
                max_roc_auc = roc_auc
                not_improving_count = 0 # Restart count as the model improved
                GE.save_weights(model)
            
            print(f"timing,{timeit.default_timer()-total_time}s,{roc_auc}", flush=True)
  
        print('[>] Model saved to file "{}"'.format(model))
        print('[>] Best ROC AUC score: {:.5f}'.format(max_roc_auc))
  
  # Testing
  ################################################################################################
    if not disable_testing:
        print('[>] Testing:')
        GE.load_weights(model) # Get best model
      
        print('\ta. Generate testing data:')
        Ys, E1s, E2s = generate_CNN_input(testi_map, no_testing_pairs)
        dataset = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(E1s), tf.data.Dataset.from_tensor_slices(E2s)))
      
        print('\tb. Calculate predictions and loss:')
        del E1s, E2s
        dataset = dataset.batch(validation_batch)
        cosine_sim = tf.concat([cosine_similarities(GE(E1s), GE(E2s)) for E1s,E2s in dataset], axis=0)
        del dataset
        difference = cosine_sim - tf.reshape(tf.dtypes.cast(Ys, tf.float32), [-1])
        squared_loss = difference * difference
        loss = tf.reduce_sum(squared_loss) / len(Ys)
        _Ys = 1.0 - (tf.math.acos(cosine_sim) / np.pi)
        print('\t\tLoss: {:.5f}'.format(loss))
      
        print('\tc. Calculate ROC curve and ROC AUC score:')
        fpr, tpr, _ = roc_curve(Ys, _Ys)
        roc_auc = auc(fpr, tpr)
        print('\t\tAUC:  {:.5f}'.format(roc_auc))
        
        # Save ROC Curve to CSV
        with open(f"roc_curve_{input}.csv","w") as buffer:
            buffer.write("fpr,tpr\n")
            for fp,tp in zip(fpr,tpr):
                buffer.write(f"{fp},{tp}\n")
      
        print('[>] Model saved to file "{}"'.format(model))
        print('[>]  AUC score: {:.5f}'.format(roc_auc))

print("Total time : ",timeit.default_timer()-total_time)
