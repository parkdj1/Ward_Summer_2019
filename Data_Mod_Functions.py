#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Data Modification Functions
    -- normalize
    -- split data into groups or subsets
    -- extract labels
    -- extract classifications

Created on Thu Jul 11 12:24:02 2019

@author: deborah
"""

import numpy as np
from copy import copy


def normalize (data):

    # calculate column means and standard deviations
    rows,cols = data.shape
    col_mean = np.mean(data[:,:-1], axis = 0, dtype = np.float64)
    col_stDev = np.std(data[:,:-1], axis = 0, dtype = np.float64)

    # normalize the data values for each column
    norm_attributes = np.array(((data[:,:-1] - col_mean) / col_stDev), dtype = np.float64)
    norm_data = np.column_stack((norm_attributes, data[:,-1]))

    # return normalized data
    return copy(norm_data)



def extract_labels (data, label_column):
    return (np.unique(data[:, label_column]))



def extract_classifications (data, label_column):
    # get labels
    labels = extract_labels(data, label_column)

    # create data_class matrix
    classifiations = np.zeros((data.shape[0], len(labels)))

    # assign vector value for each data class ie. (1,0,...,0), (0,...,1,...,0), (0,...,0,1)
    for i in range(data.shape[0]):
        for j in range(len(labels)):
            if data[i, label_column] == labels[j]:
                classifiations[i,j] = 1
        # raise error if data has value not in labels array
        if np.amax(classifiations[i,:]) == 0:
            raise ValueError('unable to categorize value in data class array')

    return classifiations



def split_groups (data, num_groups, label_column):
    labels = extract_labels(data, label_column)
    indices_split = [[] for i in range(num_groups)]

    for i in range(len(labels)):

        # isolate indices of each class and randomly shuffle
        indices = [j for j in range(data.shape[0]) if data[j, label_column] == labels[i]]
        indices_random = np.random.permutation(indices)
        indicies_split_i = np.array_split(indices_random,num_groups)

        # store random groups from class j
        for j in range(num_groups):
            indices_split[j].extend(list(indicies_split_i[j]))

    # sort indices
    for i in range(num_groups):
        indices_split[i] = sorted(indices_split[i])
        indices_split[i] = list(map(lambda x: int(x), indices_split[i]))


    return indices_split



def subset (data, condition):
    return copy(data[np.nonzero(condition),:])
