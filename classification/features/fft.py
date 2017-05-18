#! /usr/bin/python3
from classification.features.feature_classifier import FeatureExtractor
import numpy as np
import itertools


class FFTCoeffsExtractor(FeatureExtractor):

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
        # peaks=self.find_spectrum_peaks(spectrum)
        # sorted_peaks=sorted(peaks, key=lambda pair: -pair[1])
        # top_peaks_freqs=[peak[0] for peak in sorted_peaks][:3]
        return amplitudes

    def get_spectrum(self, signal):
        freqs=np.fft.fftfreq(len(signal))
        spectrum=np.fft.fftn(signal)
        magnitudes=abs(spectrum)
        up_to=len(freqs)//2
        #joined=list(zip(freqs[1:up_to], magnitudes[1:up_to]))
        joined=list(zip(freqs[1:up_to], magnitudes[1:up_to]))
        return joined

    def find_spectrum_peaks(self, spectrum):
        freqs, magnitudes=zip(*spectrum)
        def is_peak(ind):
            cur=magnitudes[ind]
            left=magnitudes[ind-1]
            right=magnitudes[ind+1]
            diff_left=cur-left
            diff_right=cur-right
            threshold=cur/40
            return ((diff_left>threshold and diff_right>threshold)
                    or (diff_left<-threshold and diff_right<-threshold))
        extrems_at=[ind-1 for ind in range(1, len(freqs)-1)
                    if is_peak(ind)]
        top_spectrum=[spectrum[int(ind)] for ind in extrems_at]
        return top_spectrum
