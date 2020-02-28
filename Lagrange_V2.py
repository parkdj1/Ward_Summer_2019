#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 10:42:23 2019

@author: deborah
"""

import numpy as np
import pandas as pd
import scipy
import scipy.spatial
from copy import copy
import itertools as it


def main():

    # Load dataset from website to array and normalize data
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    column_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

    # read in data and store in an array
    data_frame = pd.read_csv(url, names = column_names)
    array = data_frame.values
    norm_data = []
    norm_data = normalized_data(array, norm_data)


    # calculate distance between all points pair-wise
    weights = np.array((0.25, 0.3, 0.75, 1), dtype = np.float64)
    #distance_metric = lambda u, v: np.sqrt(((np.multiply(((u-v)**2), weights)).sum()))
    distance_metric = lambda u, v: np.sqrt(sum((u-v)**2*weights))
    distances = scipy.spatial.distance.pdist(norm_data[:,:-1], distance_metric)
    index_mapping = list(it.combinations(range(norm_data.shape[0]),2)) # mapping for values in pdist

    # split data and get predictions
    num_groups = 5
    num_iterations = 8
    labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
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
    degree_bounds = [5,10]
    laplacian = find_laplacian_matrix(distances,  degree_bounds,norm_data.shape[0], index_mapping)


    # use each group as test group once each; use all other groups for training
    for l in range(num_iterations):

        # Split data into groups for cross validation
        split_indices = split_data_cross(copy(norm_data), labels, train_percent, num_groups)
        iteration_accuracy = []

        for j in range(num_groups):

            # Split data into training/testing groups
            test_indices = copy(split_indices[j,:])
            train_indices = copy(np.delete(copy(split_indices), j, 0).flatten())

            # use nearest neighbors to predict tags for test group
            predictions = get_predictions(train_indices, test_indices, copy(data_class), copy(distances), degree_bounds, index_mapping,laplacian)

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


def find_laplacian_matrix(distances, degree_bounds, num_rows, index_mapping):

    laplacian_matrix = np.zeros((num_rows,num_rows),dtype = np.float64)
    degrees = [0 for i in range(num_rows)] # set initial degree of each vertex to 0
    components = np.array([0 for i in range(num_rows)],dtype = int) # set initial component value to 0
    min_degree = degree_bounds[0]
    max_degree = degree_bounds[1]
    max_dist = copy(np.amax(distances))

    k = 0
    while  np.amin(degrees) < min_degree:
        k += 1
        ind = np.argmin(distances) # find smallest distancce
        i,j = index_mapping[ind] # i,j is the location in the Laplacian corresponding to location in distances

        cond1 = degrees[i] < max_degree
        cond2 = degrees[j] < max_degree

        if cond1 and cond2:
            degrees[i] += 1
            degrees[j] += 1

            if distances[ind] > 0:
                val = 1/copy(distances[ind])
            else:
                val = 10**4 + 1

            laplacian_matrix[i,j] = -val
            laplacian_matrix[j,i] = -val
            laplacian_matrix[i,i] += val
            laplacian_matrix[j,j] += val

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

            else: # already same component
                continue

        distances[ind] = max_dist + 1

    return laplacian_matrix

def get_predictions(train_indices, test_indices, data_class, distances, degree_bounds, index_mapping, laplacian):

    num_classes = data_class.shape[1]
    num_unknown = len(test_indices)
    x = np.zeros((num_unknown, num_classes))

    # Set up matrix equation Ax = b to solve for x

    # split laplacian into L1 (columns known) and L2 (columns unknown)
    L1 = laplacian[:,train_indices.astype(int)]
    L2 = laplacian[:,test_indices.astype(int)]

    # build matrix A
    A = L2

    # solve system for each dimension (number of classes of data)
    for i in range(num_classes):
        # build matrix b
        b = -np.dot(L1, data_class[train_indices.astype(int),:][:,i])

        # Solve the system
        x[:,i] = scipy.linalg.lstsq(A.astype(int), b.astype(int))[0]

    # assign class to each data value using calculations
#    previous_norm = 0
    predictions = np.zeros(num_unknown)
    # loop through each test data point
    for i in range(num_unknown):
        calculated_value = x[i,:]
#        # loop through each class to calculate the norm for the current calculated value
#        for j in range(num_classes):
#            class_value = np.zeros(num_classes)
#            class_value[j] = 1
#            # calculate the L2 norm bewteen the prediction and each class value
#            current_norm = np.linalg.norm(calculated_value-class_value)
#            # use the smallest norm value to determine the classification
#            if current_norm <= previous_norm:
#                predictions[i] = j
#            previous_norm = current_norm
        predictions[i] = np.argmax(calculated_value)

    return predictions


def get_accuracy(test_indices, data, labels, predictions):


    count = 0

    for i in range(len(test_indices)):
        prediction = labels[predictions.astype(int)[i]]
        actual = data[test_indices[i],-1]
        if prediction == actual:
            count += 1
        else:
            print('Missclassified:',i,prediction, actual)

    return(count/len(test_indices)*100)



if __name__== "__main__":
    main()

