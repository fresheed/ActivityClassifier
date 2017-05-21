from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from collections import Counter
import numpy as np
import pandas as pd
from classification.preparation import get_classified_chunks, split_items_set
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


class Experiment(object):
    chunk_duration_seconds=3

    def run(self, log_dir, classes):
        chunk_duration=pd.to_timedelta("%ds" % Experiment.chunk_duration_seconds)
        classified_chunks=get_classified_chunks(log_dir, classes, 
                                                chunk_duration)

        train_set, test_set=split_items_set(classified_chunks)
        display_chunks_stats(classes, train_set, test_set)

        transformer=self.transformer
        transformer_params={"transformer__%s" % key: value
                            for key, value in self.transformer_params.items()}
        classifier=self.classifier
        classifier_params={"classifier__%s" % key: value
                           for key, value in self.classifier_params.items()}
        pipeline=Pipeline(steps=(
            ("transformer", transformer),
            ("classifier", classifier),
        ))

        params=dict(transformer_params)
        params.update(classifier_params)
        searcher=GridSearchCV(pipeline, param_grid=params)

        train_items, train_classes=zip(*train_set)
        test_items, test_classes=zip(*test_set)
        searcher.fit(train_items, train_classes)
        classified=searcher.predict(test_items)
        confmat=ConfusionMatrix(test_classes, classified)
        display_accuracy(confmat)
        print("Best params:", searcher.best_params_)


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
