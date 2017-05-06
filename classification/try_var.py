#! /usr/bin/python3
from core.signal import Signal
import os
import pandas as pd
import matplotlib
matplotlib.use('TkAgg') 
import pylab as plt
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
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

    #aclr_x=even_frame["x"]    
    aclr_x=even_frame
    seria_len=len(aclr_x)

    train_seria, test_seria=aclr_x[:seria_len//2], aclr_x[seria_len//2:]

    model = VAR(train_seria)
    # orders_by_criterias = model.select_order()
    # order = orders_by_criterias['fpe']
    order=3
    
    model_fit = model.fit(order)
    print("order:", order)
    print(model_fit.coefs)
    
    predictions=model_fit.forecast(even_frame.values[-order:], len(test_seria))
    
    for axis in range(3):
        plt.subplot(3, 1, axis+1)
        plt.plot(test_seria.index[:100], predictions[:, axis][:100], label="predictions")
        plt.plot(test_seria.iloc[:100, axis], label="expected")
        plt.legend(loc="upper right")
    plt.show()
    
