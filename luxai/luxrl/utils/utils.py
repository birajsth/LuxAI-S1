import numpy as np


def slice_array(arr, slicing_lengths):
    slices = []
    start = 0
    for length in slicing_lengths:
        slices.append(arr[start:start+length])
        start += length
    return slices

def merge_actions(a, b):
    return [ [a[i]]+[b[i]] for i in range(len(a))]

def index_nested_list(ids_list):
    nested_list = []
    for id in set(ids_list):
        indices = [i for i, x in enumerate(ids_list) if x == id]
        nested_list.append(indices)
    return nested_list




