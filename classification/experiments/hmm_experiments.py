from classification.experiments.experiments import Experiment
from classification.features import hmm
from sklearn.naive_bayes import GaussianNB


class HMMExperiment(Experiment):
    transformer=hmm.HMMCoeffsExtractor()
    transformer_params={}
    classifier=GaussianNB()
    classifier_params={}

 
if __name__=="__main__":
    log_dir=("/home/fresheed/research/diploma"
             "/ActivityClassifier/parse/parsed_logs/")
    classes=["pushups5_", "walk50_", "sits10_", "typing_"]
    HMMExperiment().run(log_dir, classes)
