
from classification.metric.metric_classifier import MetricClassifier, TrainedModel, compute_metric_table
from sklearn import neighbors


class KNNClassifier(MetricClassifier):

    def train(self, train_items, train_classes):
        distances=compute_metric_table(self.metric, train_items,
                                       train_items)
        classifier=neighbors.KNeighborsClassifier(5, metric="precomputed")
        classifier.fit(distances, train_classes)
        return KNNTrainedModel(self.metric, classifier, train_items)


class KNNTrainedModel(TrainedModel):

    def __init__(self, metric, fitted_model, train_items):
        self.fitted_model=fitted_model
        self.train_items=train_items
        super(KNNTrainedModel, self).__init__(metric)

    def classify(self, test_items):
        distances=compute_metric_table(self.metric, test_items, 
                                       self.train_items)
        results=self.fitted_model.predict(distances)
        return results

        
        
        
