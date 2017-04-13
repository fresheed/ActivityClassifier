#! /usr/bin/python3
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import pylab as plt


def parse_by_columns(csv_path):
    data=np.genfromtxt(csv_path, delimiter=" ", 
                       dtype=None)

    min_time=data[0][0]
    time=[entry[0]-min_time for entry in data]
    #print(time)
    val_x=[entry[1] for entry in data]
    val_y=[entry[2] for entry in data]
    val_z=[entry[3] for entry in data]

    return (time, val_x, val_y, val_z)

def plot_multidimensional(time, axes_data):
    n_rows=len(axes_data)
    n_cols=1

    for num in range(0, len(axes_data)):
        plt.subplot(n_rows, n_cols, num+1)
        plt.plot(time, axes_data[num])
        plt.xticks(np.linspace(min(time), max(time), num=20))
    

def main():
    logs_dir="parsed_logs"
    if sys.argv[1]:
        select_criteria=lambda arg: sys.argv[1] in arg
    else:
        select_criteria=lambda arg: True
    files=[os.path.join(logs_dir, log) 
           for log in filter(select_criteria, os.listdir(logs_dir))]
    for path in files:
        plt.figure(num=path)
        time, val_x, val_y, val_z=parse_by_columns(path)
        plot_multidimensional(time, (val_x, val_y, val_z))
    plt.show()


main()
