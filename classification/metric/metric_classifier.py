#! /usr/bin/python3
import numpy as np


class MetricClassifier(object):

    def __init__(self, metric):
        self.metric=metric
    

class TrainedModel(object):

    def __init__(self, metric):
        self.metric=metric
    
    def classify(self, test_items):
        pass


def compute_metric_table(metric, args_x, args_y):
    table=np.ndarray((len(args_x), len(args_y)))
    for ix, x in enumerate(args_x):
        for iy, y in enumerate(args_y):
            table[ix, iy]=metric(x, y)
    return table



        
