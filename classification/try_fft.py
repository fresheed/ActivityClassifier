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
    magnitudes=abs(spectrum[:, 0])
    plt.plot(freqs, magnitudes)
    

    print("refactor it!")
    up_to=len(freqs)//2
    joined=list(zip(freqs[1:up_to], magnitudes[1:up_to]))
    print("joined:", list(joined))
    sorted_spectrum=sorted(joined, key=lambda item: (-1)*item[1])
    print("most significant freqs:", sorted_spectrum[:5])
    top_freqs=[spc[0] for spc in sorted_spectrum[:5]]

    
    plt.show()
