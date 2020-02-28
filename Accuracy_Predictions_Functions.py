#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Accuracy and Prediction Functions
-- get predictions
-- calculate accuracy

Created on Sun Jul 14 17:09:16 2019

@author: deborah
"""

import scipy
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def get_predictions_Lap (test_indices, train_indices, classifications, laplacian, labels):
    num_rows = classifications.shape[0]
    num_classes = classifications.shape[1]
    num_unknown = len(test_indices)
    x = np.zeros((2*num_rows - num_unknown, num_classes))

    # Set up matrix equation Ax = b to solve for x
    # build matrix A
    A = np.zeros((2*num_rows - num_unknown, 2*num_rows - num_unknown))
    A[:num_rows, :num_rows] = 2 * np.dot(laplacian, laplacian)
    A01 = np.zeros((num_rows, (num_rows - num_unknown)))
    for i in range(len(train_indices)):
        A01[train_indices[i],i] = 1
    A[:num_rows, num_rows:] = A01
    A[num_rows:, :num_rows] = copy(A01).transpose()
    b = np.zeros(2*num_rows - num_unknown)

    # solve for x for each class (ie. number of dimensions of classification)
    for i in range(num_classes):
        # build matrix b
        b[num_rows:] = classifications[train_indices.astype(int),:][:,i]

        # Solve the system
        x[:,i] = scipy.linalg.lstsq(A, b)[0]

    # assign class to each data value using calculations
    predictions = np.empty(num_unknown, dtype = object)
    # loop through each test data point
    for i in range(num_unknown):
        calculated_value = x[test_indices[i],:]
        predictions[i] = labels[np.argmax(calculated_value)]

    return predictions



def get_predictions_reduced_Lap (test_indices, train_indices, classifications, laplacian, labels):

    num_classes = classifications.shape[1]
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
        b = -np.dot(L1, classifications[train_indices.astype(int),:][:,i])

        # Solve the system
        x[:,i] = scipy.linalg.lstsq(A, b)[0]

    # assign class to each data value using calculations
    predictions = np.empty(num_unknown, dtype = object)
    # loop through each test data point
    for i in range(num_unknown):
        calculated_value = x[i,:]
        # loop through each class to calculate the norm for the current calculated value
        norms = np.zeros((num_classes,),dtype = np.float64)
        for j in range(num_classes):
            vect = copy(calculated_value)
            vect[j] -= 1

            # calculate the L2 norm bewteen the prediction and each class value
            norms[j] = np.linalg.norm(vect)

        # use the smallest norm value to determine the classification
        predictions[i] = labels[np.argmin(norms)]

    return predictions


def get_predictions_nn (test_indices, tree, data_class, labels):

   # assign class to each data value using calculations
    predictions = []

    for i in test_indices:
        current_val = [0,0,0]
        # for each neighbor add class value to array
        for k in range(len(tree[i,:])):
            if tree[int(i),int(k)] != 0:
                current_val += data_class[int(k),:]
        # prediction is class of highest frequency among neighbors
        predictions.append(labels[np.argmax(current_val)])

    return np.asarray(predictions)



def calculate_accuracy_num (test_indices, predictions, data, labels):

    count = 0

    for i in range(len(test_indices)):
        prediction = labels[predictions.astype(int)[i]]
        actual = data[test_indices[i],-1]
        if prediction == actual:
            count += 1
        else:
            print('Missclassified:',i,prediction, actual)

    return(count/len(test_indices)*100)

def calculate_accuracy_cl (test_indices, predictions, data):

    count = 0

    for i in range(len(test_indices)):
        prediction = predictions[i]
        actual = data[test_indices[i],-1]
        if prediction == actual:
            count += 1
#        else:
#            print('Missclassified:',i,prediction, actual)

    return(count/len(test_indices)*100)



def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()



def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()



def plot_confusion_matrix(actual, predictions, title, labels, cmap = plt.cm.Blues):
    # Compute confusion matrix
    conf_matrix = confusion_matrix(actual, predictions)
    print(conf_matrix)

    precisions = np.zeros(len(labels))
    for i in range(len(labels)):
        precisions[i] = precision(i, conf_matrix)

    print(precisions)

    fig, ax = plt.subplots()
    im = ax.imshow(conf_matrix, interpolation = 'nearest', cmap = cmap)
    ax.figure.colorbar(im, ax = ax)
    # Show all ticks
    ax.set(xticks=np.arange(conf_matrix.shape[1]),
           yticks=np.arange(conf_matrix.shape[0]),
           # Label ticks
           xticklabels = labels, yticklabels = labels,
           title = title,
           ylabel='Actual Classification',
           xlabel='Predicted Classification')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt =  'd'
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, format(conf_matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    fig.tight_layout()

    return ax
