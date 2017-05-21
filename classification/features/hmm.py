#! /usr/bin/python3
from classification.features.feature_extraction import LogFeatureExtractor
from hmmlearn import hmm
import warnings


class HMMCoeffsExtractor(LogFeatureExtractor):

    def extract_item_features(self, item):
        num_states=3
        init=hmm.GaussianHMM(num_states, covariance_type="full",
                             algorithm="map",)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted=init.fit(item,)
        #raise ValueError("stop")
        params=fitted.covars_.flatten()

        # init=hmm.GaussianHMM(num_states, covariance_type="diag",
        #                      algorithm="map",)
        # fitted=init.fit(item,)
        # params=np.hstack([np.diagonal(mtx) for mtx in fitted.covars_])
        return params
