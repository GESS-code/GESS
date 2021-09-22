import numpy as np
import random
from progress.bar import IncrementalBar
import tensorflow as tf
from patchy_san import get_CNN_input

def parse_filename(filename, no_file_endings=1):
    """
    Parse information from file names.

    Args:
        filename (str):             Name of the file to parse
        includes_extension (bool):  Indication whether file name includes file extension or not
        no_file_endings (int):      Indication of how many endings the file has (e.g. 2 for ".train.npz")
    Returns:
        dict:                       Parsed information
    """
    from os import path
    import re

    # Remove file extenion(s)
    for _ in range(no_file_endings):
        filename = path.splitext(filename)[0]

    # Split file name with separator character `-` (do not split `--`, escaped separator character)
    file_split = re.split(r'(?<!-)-{1}(?!-)', filename)

    # Parse compiler flag optimization level
    def parse_compiler_optimization_level(compiler_flags_raw):
        compiler_optimization_level = re.search(r'(O[0-3s])', compiler_flags_raw)
        if not compiler_optimization_level: return
        compiler_optimization_level = compiler_optimization_level.group(1)
        if not compiler_optimization_level: return
        return compiler_optimization_level

    # Full information
    file_info = {}
    if len(file_split) == 10:
        file_info['Timestamp'] = file_split[0]
        file_info['Program'] = {
            'Name': file_split[1],
            'Version': file_split[2],
            'ObjectFile': file_split[3]
        }
        file_info['Architecture'] = file_split[4]
        file_info['Compiler'] = {
            'Name': file_split[5],
            'Version': file_split[6],
            'Flags': {
                'Raw': file_split[7]
            }
        }
        compiler_optimization_level = parse_compiler_optimization_level(file_split[7])
        if compiler_optimization_level: file_info['Compiler']['Flags']['OptimizationLevel'] = compiler_optimization_level
        file_info['Function'] = {
            'Offset': file_split[8],
            'Name': file_split[9]
        }
    # Missing timestamp and function information
    elif len(file_split) == 7:
        file_info['Program'] = {
            'Name': file_split[0],
            'Version': file_split[1],
            'ObjectFile': file_split[2]
        }
        file_info['Architecture'] = file_split[3]
        file_info['Compiler'] = {
            'Name': file_split[4],
            'Version': file_split[5],
            'Flags': {
                'Raw': file_split[6]
            }
        }
        compiler_optimization_level = parse_compiler_optimization_level(file_split[6])
        if compiler_optimization_level: file_info['Compiler']['Flags']['OptimizationLevel'] = compiler_optimization_level
    return file_info


def get_function_info(filename, no_file_endings=1):
    """
    Extract source-level function information from an ACFG filename.

    Args:
        filename (str):         ACFG filename
        no_file_endings (int):  Indication of how many endings the file has (e.g. 2 for ".train.npz")
    Returns:
        str:                    Source-level function information
    """
    fileinfo = parse_filename(filename, no_file_endings)
    return '{}-{}-{}-{}'.format(fileinfo['Program']['Name'],
                                fileinfo['Program']['Version'],
                                fileinfo['Program']['ObjectFile'],
                                fileinfo['Function']['Name'])


def get_function_map(filenames, no_file_endings=1):
    """
    Returns a dictionary with source-level functions as keys. Values contain a list of
    corresponding filename indices.

    Args:
        filenames (list:str):   List of ACFG filenames
        no_file_endings (int):  Indication of how many endings the file has (e.g. 2 for ".train.npz")
    Returns:
        dict:                   Function mapping
    """
    mapping = {}
    for index, filename in enumerate(filenames):
        funcinfo = get_function_info(filename, no_file_endings)
        if not funcinfo in mapping: mapping[funcinfo] = set()
        mapping[funcinfo].add(index)
    return mapping


def get_chunks(lst, n):
    """
    Yield successive `n`-sized chunks from list `lst`.

    Args:
        lst (list): List to be chunked
        n (int):    Chunk size
    Returns:
        index:      Index of first item in chunk
        generator:  List chunks
    """
    for i in range(0, len(lst), n):
        yield i, lst[i:i+n]


# Generate dataset for training, validation or testing
def generate_dataset(_files, no_pairs=1):
    Ys = []     # Labels
    X1s = []    # Features matrices ACFGs 1
    A1s = []    # Adjacency matrices ACFGs 1
    X2s = []    # Feature matrices ACFGs 2
    A2s = []    # Adjacency matrices ACFGs 2
  
    if len(_files) <= 0:
        return Ys, X1s, A1s, X2s, A2s
  
    with IncrementalBar(' ', max=len(_files), suffix='%(percent)d%%') as bar:
        for function, acfgs in _files.items():
            number_of_similar = len(acfgs)
            for index, acfg in enumerate(acfgs):
                for _ in range(no_pairs):
                    similar = index
                    while similar == index:
                        similar = int(number_of_similar*random.random())
                    similar = acfgs[similar]
                    Ys += [1]
                    X1s += [np.array(acfg['X'])]
                    A1s += [np.array(acfg['A'])]
                    X2s += [np.array(similar['X'])]
                    A2s += [np.array(similar['A'])]
          
                    different_fun = function
                    while different_fun == function:
                        different_fun = random.choice(list(_files.keys()))
                    different = random.choice(_files[different_fun])
          
                    Ys += [-1]
                    X1s += [np.array(acfg['X'])]
                    A1s += [np.array(acfg['A'])]
                    X2s += [np.array(different['X'])]
                    A2s += [np.array(different['A'])]
        
            bar.next()
    return Ys, X1s, A1s, X2s, A2s
  

# Generate dataset for training, validation or testing
def generate_CNN_input(_files, no_pairs=1):
    """
    Generates the dataset from the Patchy-San embeddings of each graph
    """

    Ys = []     # Labels
    E1s = []    # Embeddings of ACFGs 1
    E2s = []    # Embeddings of ACFGs 2

    if len(_files) <= 0:
        return Ys, E1s, E2s

    for function, acfgs in _files.items():
        number_of_similar = len(acfgs)
        for index, acfg in enumerate(acfgs):
            for _ in range(no_pairs):
                similar = index
                while similar == index:
                    similar = int(number_of_similar*random.random())
                similar = acfgs[similar]
                Ys += [1]
                E1s += [acfg]
                E2s += [similar]

                different_fun = function
                while different_fun == function:
                    different_fun = random.choice(list(_files.keys()))
                different = random.choice(_files[different_fun])

                Ys += [-1]
                E1s += [acfg]
                E2s += [different]

    return Ys, E1s, E2s
   

def get_PS_embeddings(_files, params):
    """
    Computes the Patchy-San embeddings for all functions in _files
    """
    CNN_input = {}
    for function, acfgs in _files.items():
        CNN_input[function] = list(map(lambda acfg: get_CNN_input(np.array(acfg['A']), np.array(acfg['X']), params),acfgs))
    return CNN_input


def get_PS_embeddings_flat(_files, params):
    """
    Use instead of get_PS_embeddings if every dictionary entry consists of only 1 acfg
    """
    CNN_input = {}
    for name, acfg in _files.items():
        CNN_input[name] = get_CNN_input(np.array(acfg['A']), np.array(acfg['X']), params)
    return CNN_input

def generate_input_from_scratch(files, params):
    return generate_CNN_input(get_PS_embeddings(files,params))
