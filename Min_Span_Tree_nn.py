#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Min Spanning Tree w Nearest Neighbors

Created on Sun Jul 14 21:50:15 2019

@author: deborah
"""

import numpy as np
import pandas as pd
from copy import copy
import Data_Mod_Functions as dm
import Calculate_Distance as cd
import Accuracy_Predictions_Functions as ap
import Graph_Building_Functions as gb


# Load dataset from website to array and normalize data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

# read in data and store in an array
data_frame = pd.read_csv(url, names = column_names)
array = data_frame.values
norm_data = dm.normalize(array)

# extract labels from data
labels = dm.extract_labels (copy(norm_data), -1)

# calculate weights for distance metric using one-way ANOVA for each attribute
weights = cd.calculate_attribute_strength(copy(norm_data), labels, True)
print(weights)

# calculate distance between all points pair-wise
mapped_distances = cd.calculate_wminkowski_pdist (copy(norm_data), -1, weights)

# split data and get predictions
num_groups = 5
num_iterations = 8
train_percent = 0.9

# create data_class matrix
data_class = dm.extract_classifications(copy(norm_data), -1)

iteration_accuracy = []
overall_accuracy = []

# compute laplacian
degree_bounds = [3,10]
laplacian = gb.min_spanning_tree(copy(mapped_distances), degree_bounds, norm_data.shape[0])

# use each group as test group once each; use all other groups for training
for l in range(num_iterations):

    # Split data into groups for cross validation
    split_indices = dm.split_groups (copy(norm_data), num_groups, -1)
    iteration_accuracy = []

    for j in range(num_groups):

        # Split data into training/testing groups
        test_indices = copy(split_indices[j,:])
        train_indices = np.array(sorted(copy(np.delete(copy(split_indices), j, 0).flatten())))

        # use nearest neighbors to predict tags for test group
        predictions = ap.get_predictions_nn (test_indices, laplacian, copy(data_class), labels)

        # evaluate accuracy of model
        prediction_accuracy = ap.calculate_accuracy_cl (test_indices, predictions, copy(norm_data))
        iteration_accuracy = np.append(iteration_accuracy, prediction_accuracy)
    # calculate and print the average accuracy of the algorithm

    print("Average Accuracy of Test (Cross Validated) #{0}: {1:.2f}% \n".format(l+1, np.average(iteration_accuracy)))
    overall_accuracy = np.append(overall_accuracy, np.average(iteration_accuracy))

print()
print("Average Accuracy of Algorithm: {:.2f}%".format(np.average(overall_accuracy)))
