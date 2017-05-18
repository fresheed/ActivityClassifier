from classification.preparation import get_classified_chunks, split_items_set
from classification.experiments.experiments import Experiment, display_accuracy, display_chunks_stats
from classification.features import mlp, fft
import pandas as pd


class FFTExperiment(Experiment):
    
    def run(self, log_dir, classes):
        classified_chunks=get_classified_chunks(log_dir, classes, 
                                                pd.to_timedelta("3s"))

        train_set, test_set=split_items_set(classified_chunks)
        display_chunks_stats(classes, train_set, test_set)

        extractor=fft.FFTCoeffsExtractor()
        
        classifier=mlp.MLPClassifier(extractor)

        confmat=self.explore_classifier(classifier, train_set, test_set)
        display_accuracy(confmat)


if __name__=="__main__":
    log_dir=("/home/fresheed/research/diploma"
             "/ActivityClassifier/parse/parsed_logs/")
    classes=["pushups5_", "walk50_", "sits10_", "typing_"]
    FFTExperiment().run(log_dir, classes)
