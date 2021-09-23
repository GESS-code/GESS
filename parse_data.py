import os
import csv
from datetime import datetime
import math
import random
import re
from shutil import copyfile
import json
import numpy as np

"""
Parses the data into JSON format with all the arrays
Redirect stdout to a JSON file to store the parsed data
"""


## Parameters

# Folder containing the ACFGs in npz format (Damian's format)
input_dir = "PATH TO FOLDER CONTAINING ACFGS"

# Output dense matrices (for techniques like DiffPool) or just sparse ones (for techniques in pyTorch, or Patchy-San and Gemini adapted)
adjacency_matrix = False



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


def get_info_from_parsed_filename(fileinfo):
    return '{}-{}'.format(fileinfo['Program']['Name'],
                          fileinfo['Function']['Name'])
    # Reduced the identifying information -> the same function name accross different versions of a library / different objects will be considered as duplicate and removed
    #return '{}-{}-{}-{}'.format(fileinfo['Program']['Name'],
    #                            fileinfo['Program']['Version'],
    #                            fileinfo['Program']['ObjectFile'],
    #                            fileinfo['Function']['Name']) 

def get_more_from_parsed_filename(fileinfo):
    return '{}-{}'.format(fileinfo['Architecture'], fileinfo['Compiler'])

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
    return get_info_from_parsed_filename(fileinfo)



# List ACFG filenames
def get_filenames(file_ending='.npz'):
  filenames = [f for f in os.listdir(input_dir)
                if os.path.isfile(os.path.join(input_dir, f))
                and f.endswith(file_ending)]
  return filenames

train_filenames = get_filenames('.train.npz')
valid_filenames = get_filenames('.valid.npz')
testi_filenames = get_filenames('.testi.npz')

def new_map(filenames, no_file_endings):
    mapping = {}
    for filename in filenames:
        fileinfo = parse_filename(filename, no_file_endings)
        #if fileinfo["Architecture"] == "mips":Used to get only files from 1 architecture
        funcinfo = get_info_from_parsed_filename(fileinfo)
        if not funcinfo in mapping:
            mapping[funcinfo] = {}
        acfg_id = get_more_from_parsed_filename(fileinfo)
        if not acfg_id in mapping[funcinfo]:   # Check that it is not already part of the data
            data = np.load(os.path.join(input_dir, filename))
            mapping[funcinfo][acfg_id] = {
                'A' : data['A'].tolist() if adjacency_matrix else np.stack(np.nonzero(data['A'])).tolist(),
                'X' : data['X'].tolist()
            }
    return mapping

def remove_more_info(dictionary):
    """
    Removes the compilation information from the ACFGS : for each source function, transforms the map compilation option -> ACFG into a list containing only the ACFGs.
    """
    return {filename: list(other_map.values()) for filename, other_map in dictionary.items()}

# Create function maps
train_map = remove_more_info( new_map(train_filenames, 2) )
valid_map = remove_more_info( new_map(valid_filenames, 2) )
testi_map = remove_more_info( new_map(testi_filenames, 2) )

final_map = {
    "train" : train_map,
    "valid" : valid_map,
    "testi" : testi_map
}

print(json.dumps(final_map, indent=4))
