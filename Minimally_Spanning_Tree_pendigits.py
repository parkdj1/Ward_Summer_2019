#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 21:40:01 2019

@author: deborah
"""

import numpy as np
import pandas as pd
import scipy
import scipy.spatial
import scipy.stats
from copy import copy
import itertools as it
import math


def main():

    # Load dataset from website to array and normalize data
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra"
#    column_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

    # read in data and store in an array
    data_frame = pd.read_csv(url, header = None)#, names = column_names)
    array = data_frame.values
    norm_data = []
    norm_data = normalized_data(array, norm_data)

    # extract labels from data
    labels = np.unique(norm_data[:,-1])

    # calculate weights for distance metric using one-way ANOVA for each attribute
    weights = calculate_attribute_strength(norm_data, labels)
    print(weights)

    # calculate distance between all points pair-wise
    #distance_metric = lambda u, v: np.sqrt(((np.multiply(((u-v)**2), weights)).sum()))
    distance_metric = lambda u, v: np.sqrt(sum(((u-v)**2)))# * weights))
    distances = scipy.spatial.distance.pdist(norm_data[:,:-1], distance_metric)
    index_mapping = list(it.combinations(range(norm_data.shape[0]),2)) # mapping for values in pdist

    # split data and get predictions
    num_groups = 5
    num_iterations = 8
    train_percent = 0.9

    # create data_class matrix
    data_class = np.zeros((norm_data.shape[0], len(labels)))

    # assign vector value for each data class ie. (0,...,1,...,0)
    for i in range(norm_data.shape[0]):
        for j in range(len(labels)):
            if norm_data[i,-1] == labels[j]:
                data_class[i,j] = 1
        if np.amax(data_class[i,:]) == 0:
            raise ValueError('unable to categorize value in data class array')

    iteration_accuracy = []
    overall_accuracy = []

    # compute laplacian
    degree_bounds = [3,10]
    #laplacian = find_laplacian_matrix(distances,  degree_bounds,norm_data.shape[0], index_mapping)
    min_sp_tree = build_min_spanning_tree(distances,  degree_bounds,norm_data.shape[0], index_mapping)

    # use each group as test group once each; use all other groups for training
    for l in range(num_iterations):

        # Split data into groups for cross validation
        split_indices = split_data_cross(copy(norm_data), labels, train_percent, num_groups)
        iteration_accuracy = []

        for j in range(num_groups):

            # Split data into training/testing groups
            test_indices = copy(split_indices[j,:])
            train_indices = np.array(sorted(copy(np.delete(copy(split_indices), j, 0).flatten())))

            # use nearest neighbors to predict tags for test group
            predictions = get_predictions(copy(data_class), test_indices, labels, min_sp_tree)

            # evaluate accuracy of model
            prediction_accuracy = get_accuracy(test_indices, copy(norm_data), labels, predictions)
            iteration_accuracy = np.append(iteration_accuracy, prediction_accuracy)
        # calculate and print the average accuracy of the algorithm

        print("Average Accuracy of Test (Cross Validated) #{0}: {1:.2f}% \n".format(l+1, np.average(iteration_accuracy)))
        overall_accuracy = np.append(overall_accuracy, np.average(iteration_accuracy))

    print()
    print("Average Accuracy of Algorithm: {:.2f}%".format(np.average(overall_accuracy)))


def normalized_data(array, norm_data):

    # calculate column means and standard deviations
    rows,cols = array.shape
    col_mean = np.mean(array[:,:-1], axis = 0, dtype = np.float64)
    col_stDev = np.std(array[:,:-1], axis = 0, dtype = np.float64)

    # normalize the data values for each column
    norm_attributes = np.array(((array[:,:-1] - col_mean) / col_stDev), dtype = np.float64)
    norm_data = np.column_stack((norm_attributes, array[:,-1]))
    return norm_data



def calculate_attribute_strength(data, labels):
    alpha = 0.05
    stats = np.zeros(data.shape[1] - 1)
    split_data = [None]*(len(labels))

    # use the p-value from a one-way ANOVA test to determine the weight for each attribute
    for i in range(data.shape[1] - 1):
        for k in range(len(labels)):
            split_data[k] = data[[j for j in range(data.shape[0]) if data[j, -1] == labels[k]],i]
        p_value = scipy.stats.f_oneway(*split_data)[1]
        # if the p-value is very close to 0, set the weight as 1 because it is very significant
        if (p_value) < math.pow(10,-100):
            stats[i] = 1
        # if the p-value is not zero but significant, set use the p-value to determine the weight
        elif (p_value) < alpha:
            stats[i] = -math.log10(p_value)/100
        # if the p-value is not significant (greater than alpha), use a weight of 0

    return stats


def split_data_cross(data, labels, train_percent, num_groups):

    indices_split = [[] for i in range(num_groups)]

    for i in range(len(labels)):

        # isolate indices of each class and randomly shuffle
        indices = [j for j in range(data.shape[0]) if data[j, -1] == labels[i]]
        indices_random = np.random.permutation(indices)
        indicies_split_i = np.array_split(indices_random,num_groups)

        # store random groups from class j
        for j in range(num_groups):
            indices_split[j] += list(indicies_split_i[j])

    # sort indices
    for i in range(num_groups):
        indices_split[i] = sorted(indices_split[i])

    return np.array(indices_split)


def build_min_spanning_tree(distances, degree_bounds, num_rows, index_mapping):

    min_spanning_tree = np.zeros((num_rows,num_rows),dtype = np.float64)
    degrees = [0 for i in range(num_rows)] # set initial degree of each vertex to 0
    components = np.array([0 for i in range(num_rows)],dtype = int) # set initial component value to 0
    max_dist = copy(np.amax(distances))

    while  np.amin(components) == 0:
        ind = np.argmin(distances) # find smallest distancce
        i,j = index_mapping[ind] # i,j is the location in the Laplacian corresponding to location in distances

        if distances[ind] > 0:
            val = 1/copy(distances[ind])
        else:
            val = 10**4 + 1

        distances[ind] = max_dist + 1

        # add vertex to subgraphs; if loop is created, return to top of while loop
        if components[i] == 0 and components[j] == 0: # neither in a component currently, define new component
            components[i] = np.amax(components) + 1
            components[j] = copy(components[i])

        elif components[i] == 0: # only vertex j in a component, add vertex i
            components[i] = copy(components[j])

        elif components[j] == 0: # only vertex i in a component, add vertex j
            components[j] = copy(components[i])

        elif components[i] < components[j]: # both in separate components that are combined in the lower component
            components[components ==  components[j]] = copy(components[i])

        elif components[j] < components[i]: # both in separate components that are combined in the lower component
            components[components ==  components[i]] = copy(components[j])

        else: # already same component --> return to top of loop
            continue

        # no loop is created; update degree and tree matrices
        degrees[i] += 1
        degrees[j] += 1

        min_spanning_tree[i,j] = -val
        min_spanning_tree[j,i] = -val
        min_spanning_tree[i,i] += val
        min_spanning_tree[j,j] += val

        distances[ind] = max_dist + 1

    return min_spanning_tree




def get_predictions(data_class, test_indices, labels, tree):

    # assign class to each data value using calculations
    predictions = []

    for i in test_indices:
        current_val = [0,0,0]
        for k in range(len(tree[i,:])):
            if tree[int(i),int(k)] != 0:
                current_val += data_class[int(k),:]
        predictions.append(labels[np.argmax(current_val)])

    return predictions



def get_accuracy(test_indices, data, labels, predictions):

    count = 0

    for i in range(len(test_indices)):
        prediction = predictions[i]
        actual = data[test_indices[i],-1]
        if prediction == actual:
            count += 1
        else:
            print('Missclassified:',i,prediction, actual)

    return(count/len(test_indices)*100)



if __name__== "__main__":
    main()

