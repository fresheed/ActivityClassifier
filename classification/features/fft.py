#! /usr/bin/python3

import matplotlib
matplotlib.use('TkAgg') 
import pylab as plt


from classification.features.feature_classifier import FeatureExtractor
import numpy as np
import itertools


class FFTCoeffsExtractor(FeatureExtractor):

    def extract_features(self, items):
        features=[self.extract_fft_features(item) 
                  for item in items]
        #print("freqs:", features)
        print("\n refactor it! \n")
        return features

    def extract_fft_features(self, item):
        axes_coeffs=[self.process_single_axis(item[axis])
                     for axis in ("x", "y", "z")]        
        all_coeffs=list(itertools.chain.from_iterable(axes_coeffs))
        #print("all item coeffs:", all_coeffs)
        return all_coeffs

    def process_single_axis(self, item):
        freqs=np.fft.fftfreq(len(item.values))
        spectrum=np.fft.fftn(item.values)
        magnitudes=abs(spectrum)
        
        up_to=len(freqs)//2
        joined=list(zip(freqs[1:up_to], magnitudes[1:up_to]))
        sorted_spectrum=sorted(joined, key=lambda item: (-1)*item[1])
        top_freqs=[spc[0] for spc in sorted_spectrum[:5]]

        return top_freqs
