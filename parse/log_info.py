#! /usr/bin/python3
from argparse import ArgumentParser
from core.signal import Signal, display_signal
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import pylab as plt


parser=ArgumentParser()
parser.add_argument("--log", "-l", required=True)
parser.add_argument("--show", "-s", required=False, action="store_true")
args=parser.parse_args()

signal=Signal.read_3d_csv(args.log)
print("Length: %d" % len(signal.time))
time_diffs=np.diff(signal.time)
print("dt: %f..%f" % (min(time_diffs), max(time_diffs)))
if args.show:
    display_signal(signal, True)
