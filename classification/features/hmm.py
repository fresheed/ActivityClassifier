#! /usr/bin/python3
from classification.features.feature_extraction import LogFeatureExtractor
from hmmlearn import hmm
import warnings
import numpy as np


class HMMFeaturesExtractor(LogFeatureExtractor):

    def __init__(self, num_states=None, covariance_type=None):
        self.num_states=num_states
        self.covariance_type=covariance_type

    def extract_item_features(self, item):
        params=self.get_params()
        init=hmm.GaussianHMM(n_components=params["num_states"],algorithm="map", 
                             covariance_type=params["covariance_type"],)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted=init.fit(item,)

        features=self.get_hmm_features(fitted)
        return features

    def get_hmm_features(self, model):
        raise NotImplemented("Subclasses must specify it")


class HMMABExtractor(HMMFeaturesExtractor):
    
    def get_hmm_features(self, model):
        transmat=model.transmat_
        startprobs=model.startprob_
        return np.hstack([transmat.flatten(), startprobs.flatten()])


class HMMABOutExtractor(HMMFeaturesExtractor):
    
    def get_hmm_features(self, model):
        transmat=model.transmat_
        startprobs=model.startprob_
        output_means=model.means_
        return np.hstack([transmat.flatten(), startprobs.flatten(), 
                          output_means.flatten()])


class HMMOutCovarsExtractor(HMMFeaturesExtractor):
    
    def get_hmm_features(self, model):
        output_covars=model.covars_
        return output_covars.flatten()



