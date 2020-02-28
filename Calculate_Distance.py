#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Distance Calculation Functions
-- specify metric
-- weighted minkowski: provide weights

Created on Thu Jul 11 14:11:44 2019

@author: deborah
"""

import scipy
import itertools as it
import math
import numpy as np
import scipy.stats


def calculate_pairwise_dist (data, label_column, distance_metric):
    # calculate distance between all points pair-wise
    distances = scipy.spatial.distance.pdist(data[:,:label_column], distance_metric)
    index_mapping = list(it.combinations(range(data.shape[0]),2)) # mapping for values in pdist
    mapped_distances = [[distances[i],index_mapping[i][0],index_mapping[i][1]] for i in range(len(distances))]

    return sorted(mapped_distances)


def calculate_wminkowski_pdist (data, label_column, weights):
    # calculate distance between all points pair-wise
    distances = scipy.spatial.distance.pdist(data[:,:label_column], 'minkowski', p = 2, w = weights)
    index_mapping = list(it.combinations(range(data.shape[0]),2)) # mapping for values in pdist
    mapped_distances = [[distances[i],index_mapping[i][0],index_mapping[i][1]] for i in range(len(distances))]

    return sorted(mapped_distances)


def calculate_attribute_strength(data, labels, normality):
    alpha = 0.05
    stats = np.zeros(data.shape[1] - 1)
    split_data = [None]*(len(labels))

    # use the p-value from a one-way ANOVA test to determine the weight for each attribute
    for i in range(data.shape[1] - 1):
        for k in range(len(labels)):
            split_data[k] = data[[j for j in range(data.shape[0]) if data[j, -1] == labels[k]],i]
        if normality:
            p_value = scipy.stats.f_oneway(*split_data)[1]
        else:
            p_value = scipy.stats.kruskal(*split_data)[1]
        # if the p-value is very close to 0, set the weight as 1 because it is very significant
        if (p_value) < math.pow(10,-100):
            stats[i] = 1
        # if the p-value is not zero but significant, set use the p-value to determine the weight
        elif (p_value) < alpha:
            stats[i] = -math.log10(p_value)/100
        # if the p-value is not significant (greater than alpha), use a weight of 0

    return stats



def calc_attribute_strength_large(data, labels, normality, label_column):

    # split data into smaller groups
    num_groups = int(data.shape[0]/700)
    indices = [j for j in range(data.shape[0])]
    indices_random = np.random.permutation(indices)
    indices_split = np.array_split(indices_random, num_groups)

    weights = np.empty((data.shape[1]-1, num_groups))

    for i in range(num_groups):
        weight_i = calculate_attribute_strength(data[indices_split[i]], labels, True)
        weights[:,i] = weight_i

    return (np.mean(weights[1:,:], axis = 0))

