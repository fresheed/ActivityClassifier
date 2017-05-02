#! /usr/bin/python3
from argparse import ArgumentParser
from core.signal import Signal, display_signal
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import pylab as plt


class LogCutter(object):

    def __init__(self, signal):
        self.left_bound=min(signal.time)
        self.right_bound=max(signal.time)
        self.signal=signal

    def connect(self, working_figure):
        self.figure=working_figure
        self.callback=None
        self.figure.canvas.mpl_connect('button_press_event', self.onclick)
        
    def onclick(self, event):
        click_x=event.xdata
        if event.button==1 and click_x<self.right_bound:
            self.left_bound=click_x
        elif event.button==3 and click_x>self.left_bound:
            self.right_bound=click_x
        for subplot_ax in self.figure.axes:
            if len(subplot_ax.lines) > 1:
                subplot_ax.lines[-1].remove()
                subplot_ax.lines[-1].remove()
            subplot_ax.axvline(x=self.left_bound, visible=True, color="yellow")
            subplot_ax.axvline(x=self.right_bound, visible=True, color="red")
        self.figure.canvas.draw()
        
        subsignal=Signal.get_subsignal(self.signal, self.left_bound,
                                       self.right_bound)
        if self.callback:
            plt.close(self.callback)
        self.callback=plt.figure()
        plt.figure(self.callback.number)
        display_signal(subsignal, True)
        self.callback.canvas.draw()

        plt.figure(self.figure.number)
        


def main():
    parser=ArgumentParser()
    parser.add_argument("--log", "-l", required=True)
    args=parser.parse_args()

    signal=Signal.read_3d_csv(args.log)
    print("Length: %d" % len(signal.time))
    time_diffs=np.diff(signal.time)
    print("dt: %f..%f" % (min(time_diffs), max(time_diffs)))


    cutter=LogCutter(signal)
    working_figure=plt.gcf()
    display_signal(signal, False)
    #print(working_figure.number)
    cutter.connect(working_figure)

    plt.show()

    print(cutter.left_bound, cutter.right_bound)

main()
