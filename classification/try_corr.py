#! /usr/bin/python3
from core.signal import Signal
import os
import numpy as np


walking_etalon=Signal.read_3d_csv("parse/parsed_logs/walking_free_2_log.txt")


def process_class(cls, logs_dir):
    print("Processing %s" % cls)
    logs=filter(lambda arg: cls in arg, os.listdir(logs_dir))
    for log in logs:
        signal=Signal.read_3d_csv(os.path.join(logs_dir, log))
        etalon=walking_etalon
        corr_coeff=use_corr(signal, etalon)
        print("Log file %s: correlation %f" % (log, corr_coeff))

    
def use_corr(signal, etalon):
    signal_x=signal.axes_data[0]
    etalon_x=etalon.axes_data[0]
    min_length=min(len(signal_x), len(etalon_x))
    corr_coeff=np.corrcoef(signal_x[:min_length],
                           etalon_x[:min_length])
    return corr_coeff[0, 1] # corr(x, y)
