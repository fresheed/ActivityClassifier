from sklearn.metrics import confusion_matrix, f1_score
from sklearn.utils.multiclass import unique_labels
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split


class Experiment(object):

    def __init__(self, experiment_config):
        self.experiment_config=experiment_config
        self.transformer_config=experiment_config.transformer_config
        self.classifier_config=experiment_config.classifier_config

    def run(self, classified_chunks, seed=None):
        if not seed:
            seed=np.random.randint(0, 1e6)
        items, classes=zip(*classified_chunks)
        train_items, test_items, train_classes, test_classes=train_test_split(items, classes, test_size=0.3, stratify=classes, random_state=seed)

        model=self.get_optimal_model(train_items, train_classes)

        classified=model.predict(test_items)
        confmat=ConfusionMatrix(test_classes, classified)
        best_params=model.best_params_
        return ExperimentResult(confmat, best_params)

    def get_optimal_model(self, items, classes):
        searcher=self.build_optimizer()
        searcher.fit(items, classes)
        return searcher        

    def build_optimizer(self):
        pipeline=self.get_pipeline()
        params=self.get_optimization_params()
        fold_maker=StratifiedKFold(n_splits=5)
        searcher=GridSearchCV(pipeline, scoring="f1_macro", 
                              param_grid=params, cv=fold_maker,
                              n_jobs=4)
        return searcher

    def get_pipeline(self):
        transformer=self.transformer_config.estimator
        classifier=self.classifier_config.estimator
 
        pipeline=Pipeline(steps=(
            ("transformer", transformer),
            ("classifier", classifier),
        ))
        return pipeline

    def get_optimization_params(self):
        transformer_params=self.pack_params(self.transformer_config.params,
                                            "transformer")
        classifier_params=self.pack_params(self.classifier_config.params,
                                           "classifier")
        params=dict(transformer_params)
        params.update(classifier_params)
        return params

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
