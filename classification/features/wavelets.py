#! /usr/bin/python3
from classification.features.feature_extraction import LogFeatureExtractor
import itertools
import pywt
import numpy as np


class WaveletsFeaturesExtractor(LogFeatureExtractor):

    def __init__(self, wavelet_type=None):
        super(WaveletsFeaturesExtractor, self).__init__()
        self.wavelet_type=wavelet_type

    def extract_features(self, items):
        self.validate_items_length(items)
        return super(WaveletsFeaturesExtractor, self).extract_features(items)

    def validate_items_length(self, items):
        lengths=map(len, items)
        if len(set(lengths))!=1:
            raise ValueError("Current implementation only supports"
                             " same-sized items")

    def extract_item_features(self, item):
        axes_coeffs=[self.process_single_axis(item[axis])
                     for axis in ("x", "y", "z")]        
        all_coeffs=list(itertools.chain.from_iterable(axes_coeffs))
        return all_coeffs

    def process_single_axis(self, item):
        wavelet_type=self.get_params()["wavelet_type"]
        approx, details=pywt.dwt(item.values, wavelet_type)
        return np.hstack([approx, details])
