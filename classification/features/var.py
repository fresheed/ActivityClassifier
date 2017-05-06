#! /usr/bin/python3
from classification.features.feature_classifier import FeatureExtractor
from statsmodels.tsa.ar_model import AR
import itertools


class VARCoeffsExtractor(FeatureExtractor):

    def extract_features(self, items):
        features=[self.extract_var_features(item) 
                  for item in items]
        return features

    def extract_var_features(self, item):
        axes_coeffs=[self.process_single_axis(item[axis])
                     for axis in ("x", "y", "z")]
        all_coeffs=itertools.chain.from_iterable(axes_coeffs)
        return list(all_coeffs)

    def process_single_axis(self, series):
        model=AR(series)
        order=7 # was auto detected before
        model_fit=model.fit(order)
        coeffs=model_fit.params
        selected=coeffs.values[1:] # skip const weight
        return selected
        



