from classification.features import var, fft, hmm, wavelets, interpolation, raw
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from classification.metric import dtw
from sklearn import neighbors, discriminant_analysis
from enum import IntEnum


class RunContext(IntEnum):
    CI = 2
    ALL = 1


class EstimatorConfig(object):

    def __init__(self, estimator, params, run_context):
        self.estimator=estimator
        self.params=params
        self.run_context=run_context


class ExperimentConfig(object):

    def __init__(self, transformer_config, classifier_config):
        self.transformer_config=transformer_config
        self.classifier_config=classifier_config

    def __str__(self):
        name=lambda conf: conf.estimator.__class__.__name__
        return "%s -> %s" % (name(self.transformer_config),
                             name(self.classifier_config))


def get_wavelet_types():
    # manually selected top
    working_types=['sym17', 'rbio2.2', 'sym20', 'sym16', 'sym12', 'sym19',
                   'sym13', 'sym14', 'rbio3.1', 'sym18', 'sym11', 'sym8',
                   'sym15', 'sym9', 'sym10']
    return working_types


feature_transformers={
    "hmm_cov": EstimatorConfig(hmm.HMMOutCovarsExtractor(),
                               {"num_states": [3, 4],
                                "covariance_type": ["diag", "full"]},
                               RunContext.CI),
    "hmm_abo": EstimatorConfig(hmm.HMMABOutExtractor(),
                               {"num_states": [3, 4],
                                "covariance_type": ["diag", "full"]},
                               RunContext.CI),
    "fft": EstimatorConfig(fft.FFTCoeffsExtractor(),
                           {},
                           RunContext.CI),
    "var": EstimatorConfig(var.MultiARFeatureExtractor(), 
                           {"model_order": [5, 7, ]},
                           RunContext.ALL),
    "wl": EstimatorConfig(wavelets.WaveletsFeaturesExtractor(),
                          {"wavelet_type": get_wavelet_types()},
                          RunContext.CI),
    "fft_interp": EstimatorConfig(fft.SpectrumInterpolator(),
                                  {},
                                  RunContext.CI),
    "sig_interp": EstimatorConfig(interpolation.SignalInterpolator(),
                                  {},
                                  RunContext.CI),
    "stft": EstimatorConfig(fft.STFTCoeffsExtractor(),
                            {},
                            RunContext.CI),
    "raw": EstimatorConfig(raw.RawExtractor(),
                           {},
                           RunContext.CI),
}


feature_classifiers={
    "mlp": EstimatorConfig(MLPClassifier(activation="tanh"),
                           {},
                           RunContext.CI),
    "nb": EstimatorConfig(GaussianNB(),
                          {},
                          RunContext.CI),
    "lda": EstimatorConfig(discriminant_analysis.LinearDiscriminantAnalysis(),
                           {},
                           RunContext.CI),
}


metric_transformers={
    "dtw": EstimatorConfig(dtw.DTWTransformer(),
                           {},
                           RunContext.ALL)
}


metric_classifiers={
    "knn": EstimatorConfig(neighbors.KNeighborsClassifier(),
                           {"n_neighbors": [3, 5, 7]},
                           RunContext.ALL),
}


chunk_duration_seconds=1
