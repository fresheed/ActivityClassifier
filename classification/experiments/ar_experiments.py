from classification.experiments.experiments import Experiment
from classification.features import var
from sklearn.neural_network import MLPClassifier


class MultiAR_MLP(Experiment):
    transformer=var.MultiARFeatureExtractor()
    transformer_params={"model_order": [5, 7, 9]}
    classifier=MLPClassifier()
    classifier_params={}


if __name__=="__main__":
    log_dir=("/home/fresheed/research/diploma"
             "/ActivityClassifier/parse/parsed_logs/")
    classes=["pushups5_", "walk50_", "sits10_", "typing_"]
    MultiAR_MLP().run(log_dir, classes)
