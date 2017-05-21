from classification.experiments.experiments import Experiment
from classification.features import fft
from sklearn.neural_network import MLPClassifier


class FFTExperiment(Experiment):
    transformer=fft.FFTCoeffsExtractor()
    transformer_params={}
    classifier=MLPClassifier()
    classifier_params={}


if __name__=="__main__":
    log_dir=("/home/fresheed/research/diploma"
             "/ActivityClassifier/parse/parsed_logs/")
    classes=["pushups5_", "walk50_", "sits10_", "typing_"]
    FFTExperiment().run(log_dir, classes)
