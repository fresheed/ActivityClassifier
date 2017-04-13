#! /bin/python
import numpy as np


def evaluate_optimal_transform(signal, etalon):
    dists=compute_dists(signal, etalon)
    transform=compute_transform(dists)
    rows, cols=dists.shape
    cur_row, cur_col=rows-1, cols-1
    total_weight=dists[cur_row, cur_col]
    while cur_row>0 or cur_col>0:
        possible_next=[]
        if cur_col>0:
            possible_next.append((cur_row, cur_col-1))
        if cur_row>0:
            possible_next.append((cur_row-1, cur_col))
        if cur_col>0 and cur_row>0:
            possible_next.append((cur_row-1, cur_col-1))
        min_neighbor_value=min(transform[row, col] 
                               for row, col in possible_next)
        cur_row, cur_col=[(row, col) for row, col in possible_next
                          if transform[row, col]==min_neighbor_value][0]
        total_weight+=dists[cur_row, cur_col]
    return total_weight


def compute_dists(signal, etalon):
    metric=lambda one, two: abs(one-two)
    table=[[metric(signal[i], etalon[j]) 
            for i in range(len(signal))]           
           for j in range(len(etalon))]
    return np.matrix(table)


def compute_transform(dists):
    transform=np.empty(dists.shape)
    rows, cols=dists.shape
    transform[0, 0]=dists[0, 0]
    for row in range(1, rows):
        transform[row, 0]=dists[row, 0]+transform[row-1, 0]
    for col in range(1, cols):
        transform[0, col]=dists[0, col]+transform[0, col-1]
    for row in range(1, rows):
        for col in range(1, cols):
            min_neighbor=min(transform[row-1, col],
                             transform[row-1, col-1],
                             transform[row, col-1])
            transform[row, col]=dists[row, col]+min_neighbor
    return transform
