#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:29:52 2019

@author: deborah
"""

import numpy as np
import pandas as pd
from copy import copy
import matplotlib.pyplot as plt
import Data_Mod_Functions as dm
import Calculate_Distance as cd
import Accuracy_Predictions_Functions as ap
import Graph_Building_Functions as gb

def main():
    # Load dataset from website to array and normalize data
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra"
    column_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'class']
    label_column = -1

    # read in data and store in an array
    data_frame = pd.read_csv(url, names = column_names)
    array = data_frame.values
#    array = np.vstack((array[np.nonzero(array[:,label_column] == 3),:][0], array[np.nonzero(array[:,label_column] == 5),:][0], array[np.nonzero(array[:,label_column] == 6),:][0]))
    norm_data = dm.normalize(array)
    print("shape: ", norm_data.shape)

    # extract labels from data
    labels = dm.extract_labels (copy(norm_data), label_column)
    print("labels: ", labels)

    # calculate weights for distance metric using one-way ANOVA for each attribute
    weights = cd.calc_attribute_strength_large(copy(norm_data), labels, False, label_column)
    print("weights: ", weights)

    # split data and get predictions
    num_groups = 10
    num_iterations = 10

    # create data_class matrix
    data_class = dm.extract_classifications(copy(norm_data), label_column)

    iteration_accuracy = []
    overall_accuracy = []
    predictions = np.empty(0)
    actual = np.empty(0)

    # compute laplacian
    laplacian = gb.laplacian_blocks_ldb(copy(norm_data[:,:label_column]), norm_data.shape[0])

    # use each group as test group once each; use all other groups for training
    for l in range(num_iterations):

        # Split data into groups for cross validation
        split_indices = dm.split_groups (copy(norm_data), num_groups, label_column)
        iteration_accuracy = []

        for j in range(num_groups):

            # Split data into training/testing groups
            test_indices = copy(np.asarray(split_indices[j]))
            train_indices = copy(split_indices[:j])
            if j < num_groups:
                train_indices.extend(split_indices[j+1:])
            train_indices = [x for y in train_indices for x in y]
            train_indices = np.asarray(sorted(train_indices))

            # use nearest neighbors to predict tags for test group
            prediction = ap.get_predictions_reduced_Lap (test_indices, train_indices, copy(data_class), laplacian, labels)
            predictions = np.append(predictions, prediction)
            actual = np.append(actual, norm_data[test_indices, label_column])

            # evaluate accuracy of model
            prediction_accuracy = ap.calculate_accuracy_cl (test_indices, prediction, copy(norm_data))
            iteration_accuracy = np.append(iteration_accuracy, prediction_accuracy)
        # calculate and print the average accuracy of the algorithm

        print("Average Accuracy of Test (Cross Validated) #{0}: {1:.2f}% \n".format(l+1, np.average(iteration_accuracy)))
        overall_accuracy = np.append(overall_accuracy, np.average(iteration_accuracy))

    print()
    print("Average Accuracy of Algorithm: {:.2f}%".format(np.average(overall_accuracy)))

    # print confusion matrix
    ap.plot_confusion_matrix(actual.astype(str), predictions.astype(str), 'Minimal Spanning Tree by Blocks', labels.astype(str))
    plt.savefig("./Figures/Min_Tree_by_Blocks_pendig.jpeg", optimize = True, progressive = True, format = "jpeg")
    plt.show()

if __name__== "__main__":
    main()
