import datetime
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.multiclass import unique_labels
from collections import Counter
import numpy as np


class Experiment(object):
        
    def explore_classifier(self, classifier, train_set, test_set):
        train_items, train_classes=zip(*train_set)
        test_items, test_classes=zip(*test_set)

        trained_model=classifier.train(train_items, train_classes)
        classified=trained_model.classify(test_items)

        confmat=ConfusionMatrix(test_classes, classified)
        return confmat


class ConfusionMatrix(object):
    """ Utility class that holds classification results """

    def __init__(self, expected, recognized):
        self.classes=unique_labels(expected)
        self.confmat=confusion_matrix(expected, recognized,
                                      labels=self.classes)
        self.accuracy=self.compute_accuracy(self.confmat)

    def compute_accuracy(self, confmat):
        guessed=np.trace(confmat)
        total=np.sum(confmat)
        accuracy=guessed/total
        return accuracy


def display_accuracy(confmat):
    print("Confusion for %s:" % confmat.classes)
    print(confmat.confmat)
    print("Accuracy: %f" % confmat.accuracy)


def display_chunks_stats(classes, train_set, test_set):
    train_size=Counter([entry[1] for entry in train_set])
    test_size=Counter([entry[1] for entry in test_set])
    for cls in classes:
        print("%s: train %d, test %d" % (cls, train_size[cls],
                                         test_size[cls]))
