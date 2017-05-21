#! /usr/bin/python3
from classification.features.feature_extraction import LogFeatureExtractor
from statsmodels.tsa.ar_model import AR
import itertools


class MultiARFeatureExtractor(LogFeatureExtractor):
    
    def __init__(self, model_order=5):
        self.model_order=model_order

    def extract_item_features(self, item):
        axes_coeffs=[self.process_single_axis(item[axis])
                     for axis in ("x", "y", "z")]
        all_coeffs=itertools.chain.from_iterable(axes_coeffs)
        return list(all_coeffs)

    def process_single_axis(self, series):
        model=AR(series)
        model_fit=model.fit(self.model_order)
        coeffs=model_fit.params
        selected=coeffs.values[1:] # skip const weight
        return selected
