from classification.features import var, fft, hmm, wavelets, interpolation
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from classification.metric import dtw
from sklearn import neighbors, discriminant_analysis
import pywt


class EstimatorConfig(object):

    def __init__(self, estimator, params):
        self.estimator=estimator
        self.params=params


class ExperimentConfig(object):

    def __init__(self, transformer_config, classifier_config):
        self.transformer_config=transformer_config
        self.classifier_config=classifier_config


def get_wavelet_types():
    # manually selected top
    working_types=['sym17', 'rbio2.2', 'sym20', 'sym16', 'sym12', 'sym19',
                   'sym13', 'sym14', 'rbio3.1', 'sym18', 'sym11', 'sym8',
                   'sym15', 'sym9', 'sym10']
    return working_types


feature_transformers={
    "hmm_cov": EstimatorConfig(hmm.HMMOutCovarsExtractor(),
                               {"num_states": [3, 4],
                                "covariance_type": ["diag", "full"]}),
    "hmm_abo": EstimatorConfig(hmm.HMMABOutExtractor(),
                               {"num_states": [3, 4],
                                "covariance_type": ["diag", "full"]}),
    "fft": EstimatorConfig(fft.FFTCoeffsExtractor(),
                           {}),
    # removed because we need to speed up CI build
    # "var": EstimatorConfig(var.MultiARFeatureExtractor(), 
    #                        #{"model_order": [5, 7, 9]}),
    #                        {"model_order": [5, 7,]}),
    "wl": EstimatorConfig(wavelets.WaveletsFeaturesExtractor(),
                          {"wavelet_type": get_wavelet_types()}),
    "fft_interp": EstimatorConfig(fft.SpectrumInterpolator(),
                                  {}),
    "sig_interp": EstimatorConfig(interpolation.SignalInterpolator(),
                                  {}),
    "stft": EstimatorConfig(fft.STFTCoeffsExtractor(),
                            {}),
}


feature_classifiers={
    "mlp": EstimatorConfig(MLPClassifier(),
                           {}),
    "nb": EstimatorConfig(GaussianNB(),
                          {}),
    "lda": EstimatorConfig(discriminant_analysis.LinearDiscriminantAnalysis(),
                           {}),
}


metric_transformers={
    "dtw": EstimatorConfig(dtw.DTWTransformer(),
                           {})
}


metric_classifiers={
    "knn": EstimatorConfig(neighbors.KNeighborsClassifier(),
                           {"n_neighbors": [3, 5, 7]}),
}


chunk_duration_seconds=1
