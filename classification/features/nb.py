#! /usr/bin/python3
from classification.features.feature_classifier import FeatureClassifier, TrainedModel
from sklearn.naive_bayes import GaussianNB


class NaiveBayesClassifier(FeatureClassifier):

    def train(self, train_items, train_classes):
        classifier=GaussianNB()
        features=self.extractor.extract_features(train_items)
        fitted=classifier.fit(features, train_classes)
        return NaiveBayesTrainedModel(self.extractor, fitted)
    

class NaiveBayesTrainedModel(TrainedModel):

    def __init__(self, extractor, classifier):
        self.classifier=classifier
        super(NaiveBayesTrainedModel, self).__init__(extractor)
    
    def classify(self, test_items):
        features=self.extractor.extract_features(test_items)
        predictions=self.classifier.predict(features)
        return predictions
