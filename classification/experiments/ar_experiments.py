from classification.preparation import get_classified_chunks, split_items_set
from classification.experiments.experiments import Experiment, display_accuracy, display_chunks_stats, to_timedelta
from classification.features import mlp, var


class ARExperiment(Experiment):
    
    def run(self, log_dir, classes):
        classified_chunks=get_classified_chunks(log_dir, classes, 
                                                to_timedelta(3))

        train_set, test_set=split_items_set(classified_chunks)
        display_chunks_stats(classes, train_set, test_set)

        extractor=var.VARCoeffsExtractor()
        
        classifier=mlp.MLPClassifier(extractor)

        confmat=self.explore_classifier(classifier, train_set, test_set)
        display_accuracy(confmat)


if __name__=="__main__":
    log_dir=("/home/fresheed/research/diploma"
             "/ActivityClassifier/parse/parsed_logs/")
    classes=["pushups5_", "walk50_", "sits10_", "typing_"]
    ARExperiment().run(log_dir, classes)
