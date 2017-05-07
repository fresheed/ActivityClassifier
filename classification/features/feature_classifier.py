#! /usr/bin/python3


class FeatureClassifier(object):

    def __init__(self, feature_extractor):
        self.extractor=feature_extractor
    

class TrainedModel(object):

    def __init__(self, feature_extractor):
        self.extractor=feature_extractor
    
    def classify(self, test_items):
        pass


class FeatureExtractor(object):

    def extract_features(self, items):
        pass

        