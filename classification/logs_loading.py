#! /usr/bin/python3
import pandas as pd
import os
import datetime
import numpy as np
import itertools


def collect_class_logs(class_name, log_dir):

    def read_log(log_file):
        to_dt=lambda stamp: datetime.datetime.fromtimestamp(int(stamp)/1e9)
        frame=pd.read_csv(log_file, 
                          delim_whitespace=True, header=None,
                          index_col=0, encoding="utf-8-sig",
                          converters={0: to_dt},
                          names=["TS", "x", "y", "z"])
        required_period_ms=100
        even_frame=downsample(frame, required_period_ms)
        return even_frame

    to_select=lambda path: path.startswith(class_name)
    selected_files=filter(to_select, os.listdir(log_dir))
    return [read_log(os.path.join(log_dir, log_file)) 
            for log_file in selected_files]


def downsample(full_log, milliseconds):
    req_period=datetime.timedelta(milliseconds=milliseconds)
    even_frame=full_log.resample(req_period).mean().interpolate()
    return even_frame


def strip_logs(logs):
    start_cutoff=datetime.timedelta(seconds=1.5)
    end_cutoff=datetime.timedelta(seconds=1.5)    
    def strip_borders(frame):
        start, end=frame.index[0], frame.index[-1]
        if ((end-start)<(start_cutoff+end_cutoff)):
            # only cut if log is long enough
            return frame
        cut_frame=frame[(frame.index>start+start_cutoff)
                        & (frame.index<end-end_cutoff)]
        return cut_frame
    return [strip_borders(log) for log in logs]


def split_logs(logs, chunk_duration):
    return list(itertools.chain.from_iterable([get_chunks(log, chunk_duration) 
                                               for log in logs]))


def get_chunks(log, chunk_duration):
    if log.empty:
        raise InvalidLogException("Cannot split empty log")

    freq_str=pd.infer_freq(log.index)    
    period=pd.to_timedelta(pd.tseries.frequencies.to_offset(freq_str))
    log_duration=pd.Timedelta(microseconds=len(log)*period.microseconds)

    secs=lambda delta: delta.total_seconds()
    total_chunks=int(np.ceil(secs(log_duration)/secs(chunk_duration)))
    num_borders=int(total_chunks)-1
    full_chunks=int(np.floor(secs(log_duration)/secs(chunk_duration)))

    def border_for_moment(moment):
        found=np.where(log.index>=moment)[0][0]
        return found
            
    split_moments=[min(log.index)+chunk_duration*mul 
                   for mul in range(1, num_borders+1)]
    border_indices=[border_for_moment(mt) 
                    for mt in split_moments]
    chunks=np.split(log, border_indices)
    
    return chunks[:full_chunks]


def get_classified_chunks(location, classes, duration):
    def chunks_for_log(cls):
        classified_logs=collect_class_logs(cls, location)
        cut_logs=strip_logs(classified_logs)
        chunks=split_logs(cut_logs, duration)
        return chunks

    expanded_cls=lambda cls: [(chunk, cls)
                              for chunk in chunks_for_log(cls)]
    flat_by_cls=map(expanded_cls, classes)
    classified_chunks=list(itertools.chain.from_iterable(flat_by_cls))

    return classified_chunks


class InvalidLogException(Exception):
    pass
