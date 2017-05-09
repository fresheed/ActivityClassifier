import _ucrdtw as dtw
import numpy as np


def dtw_INEQUAL_TIME_metric(x, y):
    def transform_dist(x, y, row):
        index, dist = dtw.ucrdtw(x[row].values, y[row].values,
                                 0.05, False)
        return dist
    dists_for_axes=[transform_dist(x, y, row) 
                    for row in ("x", "y", "z")]
    mean_dist=np.mean(dists_for_axes)
    return mean_dist

