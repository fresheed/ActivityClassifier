#! /usr/bin/python3
from classification.features.feature_classifier import FeatureClassifier, TrainedModel
from sklearn.neural_network import MLPClassifier as MLPNet


class MLPClassifier(FeatureClassifier):

    def train(self, train_items, train_classes):
        classifier=MLPNet(activation="relu", learning_rate="adaptive",
                          hidden_layer_sizes=(100, ))
        features=self.extractor.extract_features(train_items)
        fitted=classifier.fit(features, train_classes)
        return MLPTrainedModel(self.extractor, fitted)
    

class MLPTrainedModel(TrainedModel):

    def __init__(self, extractor, classifier):
        self.classifier=classifier
        super(MLPTrainedModel, self).__init__(extractor)
    
    def classify(self, test_items):
        features=self.extractor.extract_features(test_items)
        predictions=self.classifier.predict(features)
        return predictions
