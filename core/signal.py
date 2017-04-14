import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import pylab as plt


class Signal(object):
    
    def __init__(self, time, *axes_data):
        self.time=time
        self.axes_data=axes_data
        self.dimension=len(axes_data)

    @classmethod
    def read_3d_csv(cls, log_file):
        data=np.genfromtxt(log_file, delimiter=" ", 
                           dtype=None)
        min_time=data[0][0]
        time=[entry[0]-min_time for entry in data]
        val_x=[entry[1] for entry in data]
        val_y=[entry[2] for entry in data]
        val_z=[entry[3] for entry in data]
        return Signal(time, val_x, val_y, val_z)

    @classmethod
    def save_csv(cls, signal, path):
        get_point=lambda at: ([signal.time[at], ]+
                              [axis[at] for axis in signal.axes_data]) 
        points=[get_point(index) for index in range(len(signal.time))]
        np.savetxt(path, points, delimiter=" ")

    @classmethod
    def get_subsignal(cls, original, cut_from, cut_to):
        index_from=next(filter(lambda enm: enm[1]>=cut_from, 
                               enumerate(original.time)))[0]
        index_to=next(filter(lambda enm: enm[1]>=cut_to, 
                             enumerate(original.time)))[0]
        axes_data=[data[index_from:index_to+1]
                   for data in original.axes_data]
        return Signal(original.time[index_from:index_to+1],
                      *axes_data)


def display_signal(signal, display_now=True):
    n_rows=signal.dimension
    n_cols=1

    for num in range(0, n_rows):
        plt.subplot(n_rows, n_cols, num+1)
        plt.plot(signal.time, signal.axes_data[num])
    if display_now:
        plt.show()
