#! /usr/bin/python3
from classification.metric import knn
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import os
import datetime
import numpy as np
import itertools
import _ucrdtw as dtw
from collections import Counter

def collect_class_logs(class_name):

    def read_log(log_file):
        to_dt=lambda stamp: datetime.datetime.fromtimestamp(int(stamp)/1e9)
        frame=pd.read_csv(log_file, 
                          delim_whitespace=True, header=None,
                          index_col=0, encoding="utf-8-sig",
                          converters={0: to_dt},
                          names=["TS", "x", "y", "z"])
        
        req_period=datetime.timedelta(milliseconds=100)
        even_frame=frame.resample(req_period).mean().interpolate()

        return even_frame

    log_dir=("/home/fresheed/research/diploma"
             "/ActivityClassifier/parse/parsed_logs/")
    to_select=lambda path: path.startswith(class_name)
    selected_files=filter(to_select, os.listdir(log_dir))
    return [read_log(os.path.join(log_dir, log_file)) 
            for log_file in selected_files]


def strip_logs(logs):
    start_cutoff=datetime.timedelta(seconds=1.5)
    end_cutoff=datetime.timedelta(seconds=1.5)
    def strip_borders(frame):
        start, end=frame.index[0], frame.index[-1]
        cut_frame=frame[(frame.index>start+start_cutoff)
                        & (frame.index<end-end_cutoff)]
        return cut_frame
    return [strip_borders(log) for log in logs]


def split_logs(logs):
    chunk_duration=datetime.timedelta(seconds=1.3)

    def get_chunks(log):
        log_duration=max(log.index)-min(log.index)
        num_borders=log_duration//chunk_duration

        def border_for_moment(moment):
            found=np.where(log.index>=moment)[0][0]
            return found
            
        split_moments=[min(log.index)+chunk_duration*mul 
                       for mul in range(1, num_borders+1)]
        border_indices=[border_for_moment(mt) 
                        for mt in split_moments]
        chunks=np.split(log, border_indices)
        return chunks        

    return list(itertools.chain.from_iterable([get_chunks(log) 
                                               for log in logs]))


def dtw_INEQUAL_TIME_metric(x, y):
    def transform_dist(x, y, row):
        index, dist = dtw.ucrdtw(x[row].values, y[row].values,
                                 0.05, False)
        return dist
    dists_for_axes=[transform_dist(x, y, row) 
                    for row in ("x", "y", "z")]
    mean_dist=np.mean(dists_for_axes)
    return mean_dist


def split_items_set(items):
    test_rate=0.3
    split_at=int(len(items)*(1-test_rate))
    randomized=np.random.permutation(items)
    train_set, test_set=randomized[:split_at], randomized[split_at:]
    return train_set, test_set


def get_classified_chunks(classes):
    def chunks_for_log(cls):
        classified_logs=collect_class_logs(cls)
        cut_logs=strip_logs(classified_logs)
        chunks=split_logs(cut_logs)    
        return chunks
        
    classified_chunks=[(chunk, cls)
                       for cls in classes
                       for chunk in chunks_for_log(cls)]

    return classified_chunks


def run_classifiers():
    classes=["pushups5_", "walk50_", "sits10_", "typing_1"]

    classified_chunks=get_classified_chunks(classes)

    train_set, test_set=split_items_set(classified_chunks)
    classes_size=Counter([entry[1] for entry in classified_chunks])
    train_size=Counter([entry[1] for entry in train_set])
    for cls in classes:
        print("%s: %d/%d" % (cls, train_size[cls],
                             classes_size[cls]-train_size[cls]))

    train_items, train_classes=zip(*train_set)
    test_items, test_classes=zip(*test_set)

    for classificator in [knn.KNNClassifier]:
        print("\nUsing %s" % classificator.__name__)
        classifier=classificator(dtw_INEQUAL_TIME_metric)
        trained_model=classifier.train(train_items, train_classes)
        classified=trained_model.classify(test_items)
        confmat=confusion_matrix(test_classes, classified)
        print("Confusion:")
        print(confmat)
        print("Accuracy:", accuracy_score(test_classes, classified))
    
        

if __name__=="__main__":
    #main()
    run_classifiers()
