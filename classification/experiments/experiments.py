from sklearn.metrics import confusion_matrix, f1_score
from sklearn.utils.multiclass import unique_labels
import numpy as np
from classification.preparation import split_items_set
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold


class Experiment(object):

    def __init__(self, experiment_config):
        self.experiment_config=experiment_config
        self.transformer_config=experiment_config.transformer_config
        self.classifier_config=experiment_config.classifier_config

    def run(self, classified_chunks):
        train_set, test_set=split_items_set(classified_chunks)

        transformer=self.transformer_config.estimator
        transformer_params=self.pack_params(self.transformer_config.params,
                                            "transformer")
        classifier=self.classifier_config.estimator
        classifier_params=self.pack_params(self.classifier_config.params,
                                           "classifier")
 
        pipeline=Pipeline(steps=(
            ("transformer", transformer),
            ("classifier", classifier),
        ))

        params=dict(transformer_params)
        params.update(classifier_params)

        fold_maker=StratifiedKFold(n_splits=4)
 
        searcher=GridSearchCV(pipeline, scoring="f1_macro", 
                              param_grid=params, cv=fold_maker,
                              n_jobs=4)

        train_items, train_classes=zip(*train_set)
        test_items, test_classes=zip(*test_set)
        searcher.fit(train_items, train_classes)
        classified=searcher.predict(test_items)
        confmat=ConfusionMatrix(test_classes, classified)
        best_params=searcher.best_params_
        return ExperimentResult(confmat, best_params)

    def pack_params(self, params, prefix):
        return {"%s__%s" % (prefix, key): value
                for key, value in params.items()}


class ExperimentResult(object):
    
    def __init__(self, confmat, best_params):
        self.confmat=confmat
        self.best_params=best_params


class ConfusionMatrix(object):
    """ Utility class that holds classification results """

    def __init__(self, expected, recognized):
        self.classes=unique_labels(expected)
        self.confmat=confusion_matrix(expected, recognized,
                                      labels=self.classes)
        self.accuracy=self.compute_accuracy(self.confmat)
        self.f1_score=f1_score(expected, recognized, average="macro")

    def compute_accuracy(self, confmat):
        guessed=np.trace(confmat)
        total=np.sum(confmat)
        accuracy=guessed/total
        return accuracy
