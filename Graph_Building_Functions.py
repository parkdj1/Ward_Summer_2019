#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Laplacian Graph Building Methods

Created on Sun Jul 14 17:20:17 2019

@author: deborah
"""

import numpy as np
from copy import copy
import scipy
import scipy.stats
import Calculate_Distance as cd
import Data_Mod_Functions as dm


def min_spanning_tree(distances, num_rows):

    min_spanning_tree = np.zeros((num_rows,num_rows),dtype = np.float64)
    degrees = [0 for i in range(num_rows)] # set initial degree of each vertex to 0
    components = np.array([0 for i in range(num_rows)],dtype = int) # set initial component value to 0

    while  np.amin(components) == 0:
        dist, i,j = distances[0] # i,j is the location in the Laplacian corresponding to location in distances

        if dist > 0:
            val = 1/copy(dist)
        else:
            val = 0

        distances.remove(distances[0])

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

    return np.asarray(min_spanning_tree), degrees





def laplacian_matrix(distances, degree_bounds, num_rows):

    laplacian_matrix = np.zeros((num_rows,num_rows),dtype = np.float64)
    degrees = [0 for i in range(num_rows)] # set initial degree of each vertex to 0
    components = np.array([0 for i in range(num_rows)],dtype = int) # set initial component value to 0
    min_degree = degree_bounds[0]
    max_degree = degree_bounds[1]

    while  np.amin(degrees) < min_degree:
        dist, i,j = distances[0] # i,j is the location in the Laplacian corresponding to location in distances
#            i,j = index

        cond1 = degrees[i] < max_degree
        cond2 = degrees[j] < max_degree

        if cond1 and cond2:
            degrees[i] += 1
            degrees[j] += 1

            if dist > 0:
                val = 1/copy(dist)
            else:
                val = 10**4 + 1

            laplacian_matrix[i,j] = -val
            laplacian_matrix[j,i] = -val
            laplacian_matrix[i,i] += val
            laplacian_matrix[j,j] += val

        distances.remove(distances[0])

    print(np.unique(components))

    return laplacian_matrix



def norm_laplacian_matrix(distances, degree_bounds, num_rows):

    laplacian_matrix = np.zeros((num_rows,num_rows),dtype = np.float64)
    degrees = [0 for i in range(num_rows)] # set initial degree of each vertex to 0
    min_degree = degree_bounds[0]
    max_degree = degree_bounds[1]
#    max_dist = copy(np.amax(distances))

    while  np.amin(degrees) < min_degree:
        dist, i,j = distances[0] # i,j is the location in the Laplacian corresponding to location in distances
#            i,j = index

        cond1 = degrees[i] < max_degree
        cond2 = degrees[j] < max_degree

        if cond1 and cond2:
            degrees[i] += 1
            degrees[j] += 1

            if dist > 0:
                val = 1/copy(dist)
            else:
                val = 10**4 + 1

            laplacian_matrix[i,j] = -val
            laplacian_matrix[j,i] = -val
            laplacian_matrix[i,i] += val
            laplacian_matrix[j,j] += val

        distances.remove(distances[0])

    # normalize the laplacian
    diag = copy(laplacian_matrix[np.diag_indices_from(laplacian_matrix)])
    diag_sqrt = np.sqrt(diag)
    diag_sqrt_inv = np.array([1/k for k in diag_sqrt])

    diag_sqrt_inv_mat = np.diag(diag_sqrt_inv)

    normalized_laplacian = diag_sqrt_inv_mat @ laplacian_matrix @ diag_sqrt_inv_mat

    return normalized_laplacian

# FIX THIS #
def mst_by_class (distances, num_rows, train_indices, test_indices, labels, label_column, data):
    laplacian_matrix = np.zeros((num_rows,num_rows),dtype = np.float64)
    degrees = [0 for i in range(num_rows)] # set initial degree of each vertex to 0

    for i in range(len(labels)):
        class_indices = copy(train_indices[np.nonzero(data[train_indices, label_column] == labels[i])])
        class_tree, class_degree = min_spanning_tree(copy([distances[i] for i in class_indices]), num_rows)

        laplacian_matrix += class_tree
        degrees += class_degree

    for j in test_indices:
        for k in range(5):
            dist, x,y = min([i for i in distances if j in i])

            degrees[x] += 1
            degrees[y] += 1

            if dist > 0:
                val = 1/copy(dist)
            else:
                val = 10**4 + 1

            laplacian_matrix[x,y] = -val
            laplacian_matrix[y,x] = -val
            laplacian_matrix[x,x] += val
            laplacian_matrix[y,y] += val

            # delete min distance
            distances.remove(min([i for i in distances if j in i]))

    return laplacian_matrix



def nearest_neighbors (distances, degree_bounds, num_rows, num_neighbors):
    laplacian_matrix = np.zeros((num_rows,num_rows),dtype = np.float64)
    degrees = [0 for i in range(num_rows)] # set initial degree of each vertex to 0

    for j in range(num_rows):
        for k in range(num_neighbors):
            dist, x,y = min([i for i in distances if j in i])

            degrees[x] += 1
            degrees[y] += 1

            if dist > 0:
                val = 1/copy(dist)
            else:
                val = 10**4 + 1

            laplacian_matrix[x,y] = -val
            laplacian_matrix[y,x] = -val
            laplacian_matrix[x,x] += val
            laplacian_matrix[y,y] += val

            # delete min distance
            distances.remove(min([i for i in distances if j in i]))


    # normalize the laplacian
    diag = copy(laplacian_matrix[np.diag_indices_from(laplacian_matrix)])
    diag_sqrt = np.sqrt(diag)
    diag_sqrt_inv = np.array([1/k for k in diag_sqrt])

    diag_sqrt_inv_mat = np.diag(diag_sqrt_inv)

    normalized_laplacian = diag_sqrt_inv_mat @ laplacian_matrix @ diag_sqrt_inv_mat

    return normalized_laplacian


def laplacian_blocks_mst (data_attributes, weights):
    block_size = 60
    min_degree = 2 # for block0
    max_degree = 4 # for block0
    num_neighbors = 3
    num_rows = data_attributes.shape[0]
    laplacian_matrix = np.zeros((num_rows,num_rows),dtype = np.float64)


    # compute norms
    origin = np.zeros((1,data_attributes.shape[1]),dtype = np.float64)
    norms_vector = scipy.spatial.distance.cdist(data_attributes,origin,'minkowski',p=2,w=weights)
    #norms_vector = np.sqrt(np.square(norm_data[:,:-1].astype(np.float64)) @ weights)
    norms_list = [[norms_vector[i],i] for i in range(len(data_attributes))]
    norms_list_sorted = sorted(norms_list)


    #initial block - using bounded degree subgraph algorithm (could replace by min span tree)
    block0 = sorted([norms_list_sorted[i][1] for i in range(block_size)])
    #print(block0)
    degrees0 = [0 for i in range(block_size)] # set initial degree of each vertex to 0
    distances0 = cd.calculate_wminkowski_pdist(data_attributes[block0,:], None, weights)

    ind = 0
    while  np.amin(degrees0) < min_degree and ind < len(distances0):
        i = int(distances0[ind][1]) # vertices within the block
        j = int(distances0[ind][2])

        vi = block0[i] # vertices within the graph
        vj = block0[j]

        cond1 = degrees0[i] < max_degree
        cond2 = degrees0[j] < max_degree

        if cond1 and cond2:
            degrees0[i] += 1
            degrees0[j] += 1

            if distances0[ind][0] > 0:
                val = 1/copy(distances0[ind][0])
            else:
                val = 10**4 + 1

            laplacian_matrix[vi,vj] = -val
            laplacian_matrix[vj,vi] = -val
            laplacian_matrix[vi,vi] += val
            laplacian_matrix[vj,vj] += val
        ind += 1

    # all other blocks
    start_block = copy(block_size)
    current_block = copy(block0)
    while start_block < num_rows:
        end_block = min(start_block+block_size,num_rows)
        previous_block = copy(current_block)
        current_block = [norms_list_sorted[i][1] for i in range(start_block,end_block)]

        # compute distances between blocks
        XA = data_attributes[current_block,:]
        XB = data_attributes[previous_block,:]
        distances_between_blocks = scipy.spatial.distance.cdist(XA,XB,'minkowski',p=2,w=weights)

        # compute distances within current block
        pdisti =  scipy.spatial.distance.pdist(data_attributes[current_block,:],'minkowski',p=2,w=weights)
        square_dist = scipy.spatial.distance.squareform(pdisti)
        square_dist[np.diag_indices_from(square_dist)] = np.max(square_dist)+1


        # add edges
        for k in range(len(XA)):
            vi = current_block[k] # current_vertex
            for l in range(num_neighbors):
                # find closest point to each vertex in current block from previous block
                best_ind_p = np.argmin(distances_between_blocks[k,:])
                best_distance_p = distances_between_blocks[k,best_ind_p]
                np.delete(distances_between_blocks, best_ind_p)

                vj = previous_block[best_ind_p]  #best_neighbor

                if best_distance_p > 0:
                    val_p = 1/copy(best_distance_p)
                else:
                    val_p = 10**4 + 1

                laplacian_matrix[vi,vj] = -val_p
                laplacian_matrix[vj,vi] = -val_p
                laplacian_matrix[vi,vi] += val_p
                laplacian_matrix[vj,vj] += val_p

                # find close point within current block
                best_ind_c = np.argmin(square_dist[k,:])
                best_distance_c = square_dist[k,best_ind_c]
                np.delete(square_dist, best_ind_c)

                vjj = current_block[best_ind_c]

                if best_distance_c < best_distance_p and laplacian_matrix[vi,vjj] == 0:
                    if best_distance_c > 0:
                        val_c = 1/copy(best_distance_c)
                    else:
                        val_c = 10**4 + 1

                    laplacian_matrix[vi,vjj] = -val_c
                    laplacian_matrix[vjj,vi] = -val_c
                    laplacian_matrix[vi,vi] += val_c
                    laplacian_matrix[vjj,vjj] += val_c


        start_block += block_size

    #print(laplacian_matrix[:10,:10])
    # normalize the laplacian
    diag = copy(laplacian_matrix[np.diag_indices_from(laplacian_matrix)])
    diag_sqrt = np.sqrt(diag)
    diag_sqrt_inv = np.array([1/k for k in diag_sqrt])

    #diag_sqrt_inv_mat = np.diag(diag_sqrt_inv)
    normalized_laplacian = np.multiply(diag_sqrt_inv,laplacian_matrix)
    normalized_laplacian = np.multiply(diag_sqrt_inv,normalized_laplacian.transpose())

    #normalized_laplacian = diag_sqrt_inv_mat @ laplacian_matrix @ diag_sqrt_inv_mat

    return normalized_laplacian


def laplacian_blocks_ldb (data_attributes, weights):
    block_size = 60
    min_degree = 3 # for block0
    max_degree = 6 # for block0
    num_neighbors = 3
    num_rows = data_attributes.shape[0]
    laplacian_matrix = np.zeros((num_rows,num_rows),dtype = np.float64)


    # compute norms
    origin = np.zeros((1,data_attributes.shape[1]),dtype = np.float64)
    norms_vector = scipy.spatial.distance.cdist(data_attributes,origin,'minkowski',p=2,w=weights)
    #norms_vector = np.sqrt(np.square(norm_data[:,:-1].astype(np.float64)) @ weights)
    norms_list = [[norms_vector[i],i] for i in range(len(data_attributes))]
    norms_list_sorted = sorted(norms_list)


    #initial block - using bounded degree subgraph algorithm (could replace by min span tree)
    block0 = sorted([norms_list_sorted[i][1] for i in range(2*block_size)])
    #print(block0)
    degrees0 = [0 for i in range(2*block_size)] # set initial degree of each vertex to 0
    distances0 = cd.calculate_wminkowski_pdist(data_attributes[block0,:], None, weights)

    ind = 0
    while  np.amin(degrees0) < min_degree and ind < len(distances0):
        i = int(distances0[ind][1]) # vertices within the block
        j = int(distances0[ind][2])

        vi = block0[i] # vertices within the graph
        vj = block0[j]

        cond1 = degrees0[i] < max_degree
        cond2 = degrees0[j] < max_degree

        if cond1 and cond2:
            degrees0[i] += 1
            degrees0[j] += 1

            if distances0[ind][0] > 0:
                val = 1/copy(distances0[ind][0])
            else:
                val = 10**4 + 1

            laplacian_matrix[vi,vj] = -val
            laplacian_matrix[vj,vi] = -val
            laplacian_matrix[vi,vi] += val
            laplacian_matrix[vj,vj] += val
        ind += 1

    # all other blocks
    start_block = copy(block_size)
    current_block = copy(block0)
    while start_block < num_rows:
        end_block = min(start_block+block_size,num_rows)
        previous_block = copy(current_block)
        current_block = [norms_list_sorted[i][1] for i in range(start_block,end_block)]

        # compute distances between blocks
        XA = data_attributes[current_block,:]
        XB = data_attributes[previous_block,:]
        distances_between_blocks = scipy.spatial.distance.cdist(XA,XB,'minkowski',p=2,w=weights)

        # compute distances within current block
        pdisti =  scipy.spatial.distance.pdist(data_attributes[current_block,:],'minkowski',p=2,w=weights)
        square_dist = scipy.spatial.distance.squareform(pdisti)
        square_dist[np.diag_indices_from(square_dist)] = np.max(square_dist)+1


        # add edges
        for k in range(len(XA)):
            vi = current_block[k] # current_vertex
            for l in range(num_neighbors):
                # find closest point to each vertex in current block from previous block
                best_ind_p = np.argmin(distances_between_blocks[k,:])
                best_distance_p = distances_between_blocks[k,best_ind_p]
                np.delete(distances_between_blocks, best_ind_p)

                vj = previous_block[best_ind_p]  #best_neighbor

                if best_distance_p > 0:
                    val_p = 1/copy(best_distance_p)
                else:
                    val_p = 10**4 + 1

                laplacian_matrix[vi,vj] = -val_p
                laplacian_matrix[vj,vi] = -val_p
                laplacian_matrix[vi,vi] += val_p
                laplacian_matrix[vj,vj] += val_p

                # find close point within current block
                best_ind_c = np.argmin(square_dist[k,:])
                best_distance_c = square_dist[k,best_ind_c]
                np.delete(square_dist, best_ind_c)

                vjj = current_block[best_ind_c]

                if best_distance_c < best_distance_p and laplacian_matrix[vi,vjj] == 0:
                    if best_distance_c > 0:
                        val_c = 1/copy(best_distance_c)
                    else:
                        val_c = 10**4 + 1

                    laplacian_matrix[vi,vjj] = -val_c
                    laplacian_matrix[vjj,vi] = -val_c
                    laplacian_matrix[vi,vi] += val_c
                    laplacian_matrix[vjj,vjj] += val_c


        start_block += block_size

    #print(laplacian_matrix[:10,:10])
    # normalize the laplacian
    diag = copy(laplacian_matrix[np.diag_indices_from(laplacian_matrix)])
    diag_sqrt = np.sqrt(diag)
    diag_sqrt_inv = np.array([1/k for k in diag_sqrt])

    #diag_sqrt_inv_mat = np.diag(diag_sqrt_inv)
    normalized_laplacian = np.multiply(diag_sqrt_inv,laplacian_matrix)
    normalized_laplacian = np.multiply(diag_sqrt_inv,normalized_laplacian.transpose())

    #normalized_laplacian = diag_sqrt_inv_mat @ laplacian_matrix @ diag_sqrt_inv_mat

    return normalized_laplacian


def NEW_blocks_ldb (data_attributes, weights, distances):
    labels = dm.extract_labels(distances)


    #print(laplacian_matrix[:10,:10])
    # normalize the laplacian
    diag = copy(laplacian_matrix[np.diag_indices_from(laplacian_matrix)])
    diag_sqrt = np.sqrt(diag)
    diag_sqrt_inv = np.array([1/k for k in diag_sqrt])

    #diag_sqrt_inv_mat = np.diag(diag_sqrt_inv)
    normalized_laplacian = np.multiply(diag_sqrt_inv,laplacian_matrix)
    normalized_laplacian = np.multiply(diag_sqrt_inv,normalized_laplacian.transpose())

    #normalized_laplacian = diag_sqrt_inv_mat @ laplacian_matrix @ diag_sqrt_inv_mat

    return normalized_laplacian