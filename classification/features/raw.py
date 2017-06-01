#! /usr/bin/python3
from classification.features.feature_extraction import LogFeatureExtractor
import itertools


class RawExtractor(LogFeatureExtractor):

    def extract_item_features(self, item):
        axes_coeffs=[self.process_single_axis(item[axis])
                     for axis in ("x", "y", "z")]        
        all_coeffs=list(itertools.chain.from_iterable(axes_coeffs))
        print("len:", len(all_coeffs))
        return all_coeffs

    def process_single_axis(self, item):
        return item.values
