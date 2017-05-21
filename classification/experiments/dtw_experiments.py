from classification.preparation import get_classified_chunks, split_items_set
from classification.experiments.experiments import Experiment, display_accuracy, display_chunks_stats, ConfusionMatrix
from classification.metric import dtw
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn import neighbors


class DTWExperiment(Experiment):
    transformer=dtw.DTWTransformer()
    transformer_params={}
    classifier=neighbors.KNeighborsClassifier(5, metric="precomputed")
    classifier_params={"n_neighbors": [3, 5, 7]}

     
if __name__=="__main__":
    log_dir=("/home/fresheed/research/diploma"
             "/ActivityClassifier/parse/parsed_logs/")
    classes=["pushups5_", "walk50_", "sits10_", "typing_"]
    DTWExperiment().run(log_dir, classes)

