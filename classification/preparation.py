#! /usr/bin/python3
import pandas as pd
import os
import datetime
import numpy as np
import itertools
from collections import Counter


def collect_class_logs(class_name, log_dir):

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


def split_logs(logs, chunk_duration):

    def get_chunks(log):
        log_duration=max(log.index)-min(log.index)
        num_borders=log_duration//chunk_duration

        def border_for_moment(moment):
            found=np.where(log.index>=moment)[0][0]
            return found
            
        split_moments=[min(log.index)+chunk_duration*mul 
                       for mul in range(1, num_borders+1)]
        #print("split at:", split_moments)
        border_indices=[border_for_moment(mt) 
                        for mt in split_moments]
        chunks=np.split(log, border_indices)
        return chunks[:-1]

    return list(itertools.chain.from_iterable([get_chunks(log) 
                                               for log in logs]))


def split_items_set_XXX(all_items):
    test_rate=0.3
    classes_stats=lambda items: Counter([entry[1] for entry in items])
    mentioned_classes=classes_stats(all_items).keys()

    def validate_set(items_set):
        if any(classes_stats(items_set)[cls]==0
               for cls in mentioned_classes):
            raise ValueError("Invalid split algorithm")  

    def split():
        split_at=int(len(all_items)*(1-test_rate))
        randomized=np.random.permutation(all_items)
        train_set, test_set=randomized[:split_at], randomized[split_at:]
        validate_set(test_set)
        validate_set(train_set)
        return train_set, test_set

    attempts=0
    while True:
        try:
            train_set, test_set=split()
            return train_set, test_set
        except ValueError:
            attempts+=1
            if attempts >= 2:
                raise ValueError("Cannot split chunks")


def split_items_set(all_items):
    test_rate=0.3
    train_set=[]
    test_set=[]
    for cls, items in all_items.items():
        split_at=int(len(items)*(1-test_rate))
        _tmp_items = np.empty(len(items), dtype=object)
        _tmp_items[:]= items
        randomized=np.random.permutation(_tmp_items)
        train_items, test_items=randomized[:split_at], randomized[split_at:]
        train_set.extend([(item, cls) for item in train_items])
        test_set.extend([(item, cls) for item in test_items])
    return train_set, test_set        


def get_classified_chunks(location, classes, duration):
    print("\n fix chunk split algorithm ? \n")

    def chunks_for_log(cls):
        classified_logs=collect_class_logs(cls, location)
        cut_logs=strip_logs(classified_logs)
        chunks=split_logs(cut_logs, duration)
        return chunks
        
    # classified_chunks=[(chunk, cls)
    #                    for cls in classes
    #                    for chunk in chunks_for_log(cls)]
    classified_chunks={cls: chunks_for_log(cls)
                       for cls in classes}

    return classified_chunks
