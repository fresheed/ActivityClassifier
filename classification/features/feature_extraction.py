#! /usr/bin/python3
from sklearn.base import BaseEstimator


class LogFeatureExtractor(BaseEstimator):
    """
    Base class for all feature extraction algorithms used for log processing.    
    """

    def fit(self, X, Y):
        """
        No fit required for feature extraction
        """
        return self

    def transform(self, items):
        """
        Extracts features for 3d-accelerometer data
        """
        features=[self.extract_item_features(item) 
                  for item in items]
        return features

    def extract_item_features(self, item):
        raise NotImplemented("Subclass must implement it")


        
