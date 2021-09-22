import numpy as np
import tensorflow as tf

# Implementation Parameters

key_NUMBER_OF_FEATURES = "NUMBER_OF_FEATURES"
key_K = "K"
key_WIDTH = "WIDTH"
key_STRIDE = "STRIDE"
key_FEATURE_IMPORTANCE = "FEATURE_IMPORTANCE"

# Default parameters
parameters_PS = {
    key_NUMBER_OF_FEATURES : 9,
    key_K : 8,
    key_WIDTH : 8,
    key_STRIDE : 1,
    key_FEATURE_IMPORTANCE : np.transpose(np.array([-2.6847515, -1.53549782, 3.91545348, 0.47403792, 0.67614688, -2.08350192, 0.38032184, 2.67468581, 1.31875197]))
}


"""# Patchy-San Functions"""

def get_most_important_indices(importanceTensor):
  """
  Returns the indices of the nodes whose importance is the largest

  These indices sort the vector of importance values in decreasing order.
  i.e: The first index of the returned value has maximum importance

  Parameter :
    importanceTensor : Tensor assigning to each node index a measure of ohw important it is

  Returns :
    Node indices sorted by decreasing importance
  """
  
  return np.argsort(importanceTensor)[::-1]

def get_node_importance(features, params):
  """
  Returns a tensor representing the importance of each node

  This value depends on the importance given to each feature

  Parameter :
    features : Features vector, commonly reffered to as X (V,d)
    params : Algorithm parameters
  
  Returns :
    The importance tensor (V,)
  """
  nodes_importance = np.matmul(features, params[key_FEATURE_IMPORTANCE])
  return np.reshape(nodes_importance,[-1])

def get_node_sequence(index, A, X, k, importance):
  """
  Finds the sequence of nodes to feed to a CNN to represent
  a given node in the graph

  Parameters :
    node : index of the node to embed
    A : Graph Adjacency Matrix (VxV)
    X : Graph Attributes matrix (Vxd)
    k : Receptive field size

  Returns :
    Indices of attributes to feed to CNN
  """

  vertices = []
  boundary = [index]

  while len(vertices)<k and len(boundary)>0:
    importance_of_boundary = importance[np.array(boundary)]
    order = get_most_important_indices(importance_of_boundary)
    # Add closest neighbors sorted by importance
    vertices += [boundary[i] for i in order]

    # Compute new neighbors, then new boundary
    neighbors = set([i
                     for neighbor in boundary
                     for i in A[0,A[1,:]==neighbor]])

    boundary = sorted(neighbors.difference(vertices))
  return vertices[0:k]

def get_node_features(sequence, features, params):
  """
  Gives the features corresponding to a sequence of nodes.

  If the sequence has less than k values, then the features vector is padded with zeros.

  Parameters :
    sequence : Sequence of at most k nodes whose features need to be returned (<k,)
    features : Features matrix, often corresponding to 'X' (V,d)
    params : Algorithm paraameters
  
  Returns :
    Tensor containing the attributes for the k nodes, ready to be fed to a CNN
  """

  attributes = features[sequence,:]
  #attributes = np.reshape(attributes, [-1])
  if len(sequence) == params[key_K]:
    return attributes
  else:
    missing = params[key_K] - len(sequence)
    zeros = np.zeros((missing, params[key_NUMBER_OF_FEATURES]))#np.zeros(missing * params[key_NUMBER_OF_FEATURES])
    return np.concatenate((attributes, zeros))

"""# Patchy-San"""

def Receptive_Field(A, X, importance, node, params):
  """
  Computes the receptive field of a node

  Parameters :
    A : Graph Adjacency matrix (V,V)
    X : Features matrix (V,d)
    importance : Node importance array (V,)
    node : Index of the source node
    params : Algorithm parameters
  
  Returns :
    Receptive field : Features of the node and its neighbors to feed CNN
  """

  if node == -1:  # Phantom node, does not exist
    return np.zeros((params[key_K],params[key_NUMBER_OF_FEATURES]))#np.zeros((params[key_K])*params[key_NUMBER_OF_FEATURES])
  receptive_nodes = get_node_sequence(node, A, X, params[key_K], importance)
  return get_node_features(receptive_nodes, X, params)

def get_CNN_input(A, X, params):
  """
  Computes the Patchy-San representation of a graph

  Selects most important nodes, and return their receptive fields
  Uses global parameters K, WIDTH and STRIDE

  Parameters :
    A : Adjacency matrix of the graph (V x V)
    X : Node features matrix of the graph (V x d)
    params : Algorithm parameters
  
  Returns :
    Receptive field for WIDTH nodes (WIDTH x d(k+1))
  """
  
  node_importance = get_node_importance(X, params)
  most_important_nodes = get_most_important_indices(node_importance)
  nodes_basis = most_important_nodes[:params[key_WIDTH]*params[key_STRIDE]:params[key_STRIDE]]
  if len(nodes_basis) < params[key_WIDTH]:
    nodes_basis = np.concatenate((nodes_basis,np.full(params[key_WIDTH]-len(nodes_basis),-1)))
  fields_creator = np.vectorize(lambda node : Receptive_Field(A, X, most_important_nodes, node, params), signature= "()->(m,n)")
  receptive_fields = fields_creator(nodes_basis)
  return tf.convert_to_tensor(receptive_fields,dtype=tf.dtypes.float32)
