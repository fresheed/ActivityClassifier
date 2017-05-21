#! /usr/bin/python3
from classification.features.feature_extraction import LogFeatureExtractor
import numpy as np
import itertools


class FFTCoeffsExtractor(LogFeatureExtractor):

    def extract_features(self, items):
        self.validate_items_length(items)
        return super(FFTCoeffsExtractor, self).extract_features(items)

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
        spectrum=self.get_spectrum(item.values)
        amplitudes=list(zip(*spectrum))[1]
        return amplitudes

    def get_spectrum(self, signal):
        freqs=np.fft.fftfreq(len(signal))
        spectrum=np.fft.fftn(signal)
        magnitudes=abs(spectrum)
        up_to=len(freqs)//2
        #joined=list(zip(freqs[1:up_to], magnitudes[1:up_to]))
        joined=list(zip(freqs[1:up_to], magnitudes[1:up_to]))
        return joined
