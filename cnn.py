import numpy as np
import tensorflow as tf
from patchy_san import *
from progress.bar import IncrementalBar
import sys

# Default parameters
C1 = 152
SMALL_EMBEDDING_SIZE = 30
FINAL_EMBEDDING_SIZE = 25

def make_network(params = parameters_PS):
    """
    Makes a network taking as input the Patchy-San transformation described by the given parameters
    
    Parameters:
        params: Patchy-San parameters
                defaults to the default parameters_PS
    """
    model_input = tf.keras.Input(shape=(params[key_WIDTH],params[key_NUMBER_OF_FEATURES]*params[key_K]))
    per_point = tf.keras.layers.Dense(C1)(model_input)
    per_filter = tf.keras.layers.Dense(SMALL_EMBEDDING_SIZE, activation=tf.keras.activations.relu)(tf.keras.layers.Permute((2,1))(per_point))
    flat = tf.keras.layers.Flatten()(per_filter)
    output = tf.keras.layers.Dense(FINAL_EMBEDDING_SIZE)(flat)
    return tf.keras.Model(inputs=model_input, outputs=output)

def denser_network(params = parameters_PS):
    """
    Makes a network denser than the default make_network method.
    Used for testing purposes
    
    Parameters:
    params: Patchy-San parameters
            defaults to the default parameters_PS
    """
    C0 = 15
    model_input = tf.keras.Input(shape=(params[key_WIDTH],params[key_K],params[key_NUMBER_OF_FEATURES]))
    per_point = tf.keras.layers.Dense(C0)(model_input)
    per_neigh = tf.keras.layers.Dense(C1)(tf.keras.layers.Reshape((params[key_WIDTH],C0*params[key_K]))(per_point))
    per_filter = tf.keras.layers.Dense(SMALL_EMBEDDING_SIZE, activation=tf.keras.activations.relu)(tf.keras.layers.Permute((2,1))(per_neigh))
    flat = tf.keras.layers.Flatten()(per_filter)
    output = tf.keras.layers.Dense(FINAL_EMBEDDING_SIZE)(flat)
    return tf.keras.Model(inputs=model_input, outputs=output)
    
# NOTE: The network returned by make_network with paramters 152/30/25 is the best I found
GE = make_network()
GE.summary() # Print the summary of the NN


def get_embedding(cnn, A, X, PS_params):
    """
    Gets the embedding of the graph described by A and X using
    Patchy-San and network cnn

    Parameters :
        cnn : Convolutional neural network giving the embeddings
        A : Graph adjacency matrix
        Xs: Graph features matrix
        PS_params : Patchy-San parameters
    """
    graph_descriptor = get_CNN_input(A,X, PS_params)
    return cnn(graph_descriptor)

def get_embeddings(cnn, As, Xs, PS_params):
    """
    Gets the embeddings of the graphs described by As and Xs using
    Patchy-San and network cnn

    Parameters :
        cnn : Convolutional neural network giving the embeddings
        As : Graphs adjacency matrices
        Xs : Graphs features matrices
        PS_params : Patchy-San parameters
    """

    graph_descriptors = np.stack([get_CNN_input(A,X, PS_params) for A,X in zip(As,Xs)])
    return cnn(graph_descriptors)



def cosine_similarity(X, Y):
    """
    Cosine similarity.
    From Damian's implementation
    Args:
        X (Tensor):   Embedding of ACFGs 1 (p x 1)
        Y (Tensor):   Embedding of ACFGs 2 (p x 1)
    Returns:
        cosine_sim tensor (1)
    """
    
    numerator = tf.reduce_sum(X * Y)
    denominator = tf.norm(X) * tf.norm(Y)
    return tf.clip_by_value(numerator / denominator, -1, 1)



# Loss computation

@tf.function
def cosine_similarities(X, Y):
    """
    Cosine similarity.
    From Damian's implementation
    Args:
        X (Tensor):   Embedding of ACFGs 1 (batch_size x p)
        Y (Tensor):   Embedding of ACFGs 2 (batch_size x p)
    Returns:
        cosine_sim tensor (1 x batch_size)
    """

    numerator = tf.reduce_sum(X*Y, axis= 1)
    denominator = tf.norm(X, axis=1) * tf.norm(Y, axis=1)
    denominator = tf.where(tf.math.not_equal(denominator,0), denominator,1)
    similarities = numerator / denominator
    # If division by zero, we have a 0÷0 case. Say 0 in this case
    return tf.clip_by_value(similarities, -1, 1)

@tf.function
def loss_function(Ys, Y1s, Y2s):
    """
    Sum of the square of the differences between label and cosine similarity

    Parameters :
        Ys : Labels
        Y1s : Predictions for batch 1
        Y2s : Predictions for batch 2

    Returns :
        loss tensor (1 x 1)
    """
    difference = cosine_similarities(Y1s, Y2s) - tf.reshape(tf.dtypes.cast(Ys, tf.float32), [-1])
    squared_loss = difference * difference
    return tf.reduce_sum(squared_loss)



# Training

def train(model, mb_Ys, mb_X1s, mb_A1s, mb_X2s, mb_A2s, batches, params, lr=0.0001):
    """
    Trains the network using data in mini-batches

    1 invocation of this function corresponds to 1 epoch

    Parameters :
        model : model to train
        mb_Ys : True labels, by mini-batches
        mb_X1s, mb_X2s : Feature matrices, by mini-batches
        mb_A1s, mb_A2s : Adjacency matrices, by mini-batches
        params : Patchy-San parameters
        lr=0.0001 : Optimizer learning rate
    """

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    batch_loss = lambda: loss_function(Ys, get_embeddings(model,A1s,X1s,params), get_embeddings(model,A2s,X2s,params))
    variables = lambda: model.trainable_weights
    with IncrementalBar(' ', max=batches, suffix='%(percent)d%%') as bar:
        for Ys, X1s, A1s, X2s, A2s in zip(mb_Ys, mb_X1s, mb_A1s, mb_X2s, mb_A2s):
            optimizer.minimize(batch_loss, variables)
            bar.next()

            
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

@tf.function
def handle_one(Ys, E1s, E2s):
    """
    Handles only one batch
    
    Parameters:
        Ys : Labels for the batch
        E1s, E2s : Patchy-San embeddings of graphs
    """
    batch_loss = lambda: loss_function(Ys, GE(E1s), GE(E2s))
    fast_loss = tf.function(batch_loss)
    variables = lambda: GE.trainable_weights
    optimizer.minimize(fast_loss,variables)
        
def train_fast(dataset, batches, lr=0.0001):
    """
    Same as train, but takes a dataset of already computed Patchy-San embeddings (by get_CNN_input) as input
    """
    with IncrementalBar(' ', max=batches, suffix='%(percent)d%%') as bar:
        for Ys, E1s, E2s in dataset:
            tf.distribute.get_strategy().run(handle_one,args=(Ys,E1s,E2s))
            bar.next()

            
def train_specific(model, dataset, lr=0.0001):
    """
    Trains the model given as parameter
    Works like train_fast
    """
    variables = lambda: model.trainable_weights
    for Ys, E1s, E2s in dataset:
        batch_loss = lambda: loss_function(Ys, model(E1s), model(E2s))
        fast_loss = tf.function(batch_loss)
        optimizer.minimize(fast_loss,variables)
