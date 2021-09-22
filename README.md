# GESS

This repository contains the code of GESS, introduced in __Investigating Graph Embeddings Methods for Cross-Platform Binary Code Similarity Detection__. The details of the convolutional neural network are in `cnn.py`. The model can be trained and tested using `main.py`.

The implementation requires a dataset in JSON format. The given dataset can be converted to JSON using `parse_data.py`


## Parsing data to JSON format

GESS take ACFGs in JSON format, with sparse matrices. This requires to set the parameter `adjacency_matrix` in `parse_data` to **False**.
The parameter `input_dir` should contain the path to the folder containing the ACFGs.

Then `parse_data` can be run using python and will output in the dataset in JSON to the standard output. It might make sense here to redirect *stdout* to a file with JSON extension.


## Train / Test a model

Model training uses `main.py`.
You just need to specify the path to the dataset and the model file in the code (section parameters) and then run the file. If the model file does not exist, a new model will be created. Otherwise training willl be resumed.

It runs until validation AUC reaches a maximum and doesn't improve for 25 epochs.
Then it loads the best model and tests it on the testing data. It will output the ROC curve in csv format.

The parameters in the code allow to bypass training or testing.


## Compute embeddings

When given a model and JSON file as parameters, `compute_embeddings.py` will compute the embeddings of all ACFGs in the testing dataset


## Create Top-K Precision/Recall Curve

The file `similarities.py` uses the embeddings computed by `compute_embeddings.py` and can output precision and recall accross the top-K closest candidates, for K up to 200 (tunable in code).


## Train specifically to improve Top-K Precision/Recall

Using `validate_differently.py` instead of `main.py` in order to train a model will return a model with lower ROC AUC, but trained faster and with better Top-K validation and recalls.
