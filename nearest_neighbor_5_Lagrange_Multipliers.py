#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 11:00:39 2019

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
    norm_data = norm_data[:100,:]

    # calculate distance between all points pair-wise
    weights = np.array((0.25, 0.3, 0.75, 1), dtype = np.float64)
    distance_metric = lambda u, v: (np.sqrt(((np.multiply(((u-v)**2), weights)).sum())))
    distances = scipy.spatial.distance.pdist(norm_data[:,:-1], distance_metric)
    index_mapping = list(it.combinations(range(norm_data.shape[0]),2)) # mapping for values in pdist

    # split data and get predictions; iterate multiple times and calculate avg predictive quality
    num_groups = 5
    num_iterations = 8
    labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    train_percent = 0.9
    # extract label column of data and assign numerical values to different categories
    data_class = copy(norm_data[:,-1])
    for i in range(len(data_class)):
        if data_class[i] == labels[0]:
            data_class[i] = 1
        elif data_class[i] == labels[1]:
            data_class[i] = -1
        else:
            raise ValueError('unable to categorize value in data class array')

    iteration_accuracy = []
    overall_accuracy = []

    # use each group as test group once each; use all other groups for training
    for l in range(num_iterations):
        # Split data into groups for cross validation
        split_indices = split_data_cross(copy(norm_data), labels, train_percent, num_groups)
        for j in range(num_groups):
            # Split data into training/testing groups
            test_indices = copy(split_indices[j,:])
            train_indices = copy(np.delete(copy(split_indices), j, 0).flatten())
            # use nearest neighbors to predict tags for test group
            degree_bounds = [5,10]
            predictions = get_predictions(train_indices, test_indices, copy(data_class), copy(distances), degree_bounds, index_mapping)
            # evaluate accuracy of model
            prediction_accuracy = get_accuracy(test_indices, copy(data_class), predictions)
            iteration_accuracy = np.append(iteration_accuracy, prediction_accuracy)
        # calculate and print the average accuracy of the algorithm
        print("Average Accuracy of Test with Cross Validation #{0}: {1:.2f}%".format(l+1, np.average(iteration_accuracy)))
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

    indices_split = np.empty((num_groups, 1))

    for i in range(len(labels)):

        # isolate indices of each class and randomly shuffle
        indices = [j for j in range(data.shape[0]) if data[j, -1] == labels[i]]
        indices_random = np.random.permutation(indices)
        num_per_group = len(indices)/num_groups

        # store randomized indices into the specified number of groups
        array_groups = indices_random[:int(num_per_group)]

        for i in np.arange(1,num_groups):
            array_groups = np.vstack((array_groups, indices_random[int(i*num_per_group):int((i+1)*num_per_group)]))
        indices_split = np.hstack((indices_split, array_groups))

    indices_split = np.delete(indices_split, 0, 1)

    return indices_split


def find_adjacency_matrix(distances, degree_bounds, num_rows, index_mapping):

    adjacency = np.zeros((num_rows,num_rows),dtype = np.float64)
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
                val = max_dist + 1

            adjacency[i,j] = val
            adjacency[j,i] = val

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

    return adjacency, degrees


def get_laplacian(distances, num_rows, degree_bounds, index_mapping):

    # get adjacency and degree matrices
    adjacency_matrix, degrees = find_adjacency_matrix(distances, degree_bounds, num_rows, index_mapping)

    # laplacian = adjacency - degree (degrees in the diagonal)
    diag_indices = np.diag_indices(num_rows, ndim = 2)
    adjacency_matrix[diag_indices] = np.multiply(degrees, -1)
    return adjacency_matrix


def get_predictions(train_indices, test_indices, data_class, distances, degree_bounds, index_mapping):

    num_rows = len(data_class)
    num_unknown = len(test_indices)

    # Get laplacian matrix
    laplacian = get_laplacian(distances, num_rows, degree_bounds, index_mapping)

    # Set up matrix equation Ax = b to solve for x

    # build matrix A
    A00 = 2 * np.dot(laplacian, laplacian)
    A01 = np.zeros((num_rows, (num_rows - num_unknown)))
    A01[np.diag_indices((num_rows - num_unknown), ndim = 2)] = 1
    A10 = copy(A01).transpose()
    A11 = np.zeros((num_rows - num_unknown, num_rows - num_unknown))

    A = np.hstack((np.vstack((A00,A10)), np.vstack((A01,A11))))

    # build matrix b
    b = np.hstack((np.zeros(num_rows), data_class[train_indices.astype(int)])).reshape(2*num_rows - num_unknown,1)

    # Solve the system
    x = np.linalg.lstsq(A.astype(np.float64), b.astype(np.float64))[0]

    return x


def get_accuracy(test_indices, data_class, predictions):

    count = 0

    for i in range(len(test_indices)):
        prediction = predictions[int(test_indices[i])] < 0
        actual = data_class[int(test_indices[i])] < 0
        if prediction == actual:
            count +=1

    return(count/len(test_indices)*100)



if __name__== "__main__":
    main()

