#! /usr/bin/python3
from core.signal import Signal
import os
import pandas as pd
import matplotlib
matplotlib.use('TkAgg') 
import pylab as plt
import numpy as np
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error as mse
import datetime
import traces


walk_log="parse/parsed_logs/walk50_1_log.txt"
walking=Signal.read_3d_csv(walk_log)


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


if __name__=="__main__":
    to_dt=lambda stamp: datetime.datetime.fromtimestamp(int(stamp)/1e9)
    frame=pd.read_csv("parse/parsed_logs/walk50_1_log.txt", 
                      delim_whitespace=True, header=None,
                      index_col=0, encoding="utf-8-sig",
                      converters={0: to_dt},
                      names=["TS", "x", "y", "z"])

    req_period=datetime.timedelta(milliseconds=100)

    even_frame=frame.resample(req_period).mean().interpolate()

    # plt.plot(frame["x"])
    # plt.plot(even_frame["x"])
    # plt.show()

    aclr_x=even_frame["x"]    
    seria_len=len(aclr_x)

    train_seria, test_seria=aclr_x[:seria_len//2], aclr_x[seria_len//2:]
    model = AR(train_seria)
    model_fit = model.fit()

    # test_from=train_seria.index[-1]
    # test_to=test_seria.index[-1]
    predictions = model_fit.predict(start=len(train_seria)-1,
                                    end=seria_len,
                                    dynamic=False)

    plt.plot(predictions.iloc[0:100])
    plt.plot(test_seria.iloc[0:100])
    plt.show()
