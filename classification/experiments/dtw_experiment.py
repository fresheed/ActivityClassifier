from classification.preparation import get_classified_chunks, split_items_set
from classification.experiments.experiments import Experiment, display_accuracy, display_chunks_stats
from classification.metric import knn, metrics
import pandas as pd


class DTWExperiment(Experiment):
    
    def run(self, log_dir, classes):
        classified_chunks=get_classified_chunks(log_dir, classes, 
                                                pd.to_timedelta("2s"))

        train_set, test_set=split_items_set(classified_chunks)
        display_chunks_stats(classes, train_set, test_set)

        metric=metrics.dtw_INEQUAL_TIME_metric
        classifier=knn.KNNClassifier(metric)
        
        self.use_classifier(classifier, 
                            train_set, test_set)
        
    def use_classifier(self, classifier, train_set, test_set):
        print("classifier:", classifier.__class__.__name__)
        confmat=self.explore_classifier(classifier, train_set, test_set)
        display_accuracy(confmat)
    
     
if __name__=="__main__":
    log_dir=("/home/fresheed/research/diploma"
             "/ActivityClassifier/parse/parsed_logs/")
    classes=["pushups5_", "walk50_", "sits10_", "typing_"]
    DTWExperiment().run(log_dir, classes)
