#! /usr/bin/python3
import os
import pandas as pd
import matplotlib
matplotlib.use('TkAgg') 
import pylab as plt
import numpy as np
from sklearn.metrics import mean_squared_error as mse
import datetime
import traces


walk_log="parse/parsed_logs/walk50_1_log.txt"


if __name__=="__main__":
    to_dt=lambda stamp: datetime.datetime.fromtimestamp(int(stamp)/1e9)
    frame=pd.read_csv("parse/parsed_logs/walk50_1_log.txt", 
                      delim_whitespace=True, header=None,
                      index_col=0, encoding="utf-8-sig",
                      converters={0: to_dt},
                      names=["TS", "x", "y", "z"])

    req_period=datetime.timedelta(milliseconds=100)

    even_frame=frame.resample(req_period).mean().interpolate()
    
    freqs=np.fft.fftfreq(len(even_frame.values))
    spectrum=np.fft.fftn(even_frame.values)
    #spectrum=np.fft.fft(even_frame.values[:, 0])
    
    #print(spectrum[:, 0])
    print(freqs)
    plt.plot(np.fft.fftshift(freqs), np.fft.fftshift(abs(spectrum[:, 0])))
    plt.figure()
    plt.plot(freqs, abs(spectrum[:, 0]))
    plt.show()
    
