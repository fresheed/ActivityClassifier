from sklearn.base import BaseEstimator
from classification.metric.metrics import dtw_INEQUAL_TIME_metric as dtw_metric
import numpy as np


class DTWTransformer(BaseEstimator):
    """
    Should be used in pair with KNN classifier with metric="precomputed"
    Test items are cached on fitting and used later on training and prediction
    """

    def fit(self, X, Y):
        self.etalon_items=X
        return self

    def transform(self, X):
        distances=self.compute_metric_table(X, self.etalon_items)
        return distances

    def compute_metric_table(self, args_x, args_y):
        table=np.ndarray((len(args_x), len(args_y)))
        for ix, x in enumerate(args_x):
            for iy, y in enumerate(args_y):
                table[ix, iy]=dtw_metric(x, y)
        return table


