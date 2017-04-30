#! /usr/bin/python3
from core.signal import Signal
import _ucrdtw as dtw
import os


walking_chunk=Signal.read_3d_csv("parse/chunks/walking_chunk.txt")


def process_class(cls, logs_dir):
    print("Processing %s" % cls)
    logs=filter(lambda arg: cls in arg, os.listdir(logs_dir))
    for log in logs:
        signal=Signal.read_3d_csv(os.path.join(logs_dir, log))
        chunk=walking_chunk
        index, dist=use_dtw(signal, chunk)
        print("Log file %s: DTW dist %f" % (log, dist))

    
def use_dtw(signal, chunk):
    signal_x=signal.axes_data[0]
    chunk_x=chunk.axes_data[0]
    index, dist = dtw.ucrdtw(signal_x, chunk_x, 0.05, False)
    return index, dist
