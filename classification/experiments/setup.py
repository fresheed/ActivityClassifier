from classification.features import var
from classification.features import fft
from classification.features import hmm
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from classification.metric import dtw
from sklearn import neighbors


class EstimatorConfig(object):

    def __init__(self, estimator, params):
        self.estimator=estimator
        self.params=params


class ExperimentConfig(object):

    def __init__(self, transformer_config, classifier_config):
        self.transformer_config=transformer_config
        self.classifier_config=classifier_config


feature_transformers={
    "hmm": EstimatorConfig(hmm.HMMCoeffsExtractor(),
                           {}),
    "fft": EstimatorConfig(fft.FFTCoeffsExtractor(),
                           {}),
    "var": EstimatorConfig(var.MultiARFeatureExtractor(),
                           {"model_order": [5, 7, 9]})
}


feature_classifiers={
    "mlp": EstimatorConfig(MLPClassifier(),
                           {}),
    "nb": EstimatorConfig(GaussianNB(),
                          {})
}


metric_transformers={
    "dtw": EstimatorConfig(dtw.DTWTransformer(),
                           {})
}


metric_classifiers={
    "knn": EstimatorConfig(neighbors.KNeighborsClassifier(),
                          {"n_neighbors": [3, 5, 7]})
}


chunk_duration_seconds=3
