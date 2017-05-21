from classification.experiments.experiments import Experiment
from classification.features import hmm
from sklearn.naive_bayes import GaussianNB
from classification.features.features_extraction import LogFeatureExtractor
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPClassifier


# class CombinedExperiment(Experiment):
#     transformer=hmm.HMMCoeffsExtractor()
#     transformer_params={}
#     classifier=GaussianNB()
#     classifier_params={}


# class ClassifiersCombiner(BaseEstimator):
#     classifiers={
#         "mlp": MLPClassifier(),
#         "nb": GaussianNB()
#     }
    
#     def __init__(self, extractor_code="hmm", classifier_code="mlp"):
#         self.classifier=classifiers[method]

 
if __name__=="__main__":
    log_dir=("/home/fresheed/research/diploma"
             "/ActivityClassifier/parse/parsed_logs/")
    classes=["pushups5_", "walk50_", "sits10_", "typing_"]
    CombinedExperiment().run(log_dir, classes)
