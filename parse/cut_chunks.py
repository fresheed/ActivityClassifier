#! /usr/bin/python3
from argparse import ArgumentParser
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import pylab as plt
from core.signal import Signal, display_signal


def show_chunks(signal, cut_from, cut_to):
    display_signal(signal, display_now=False)
    plt.subplot(3, 1, 1)
    plt.axvline(cut_from, color="red")
    plt.axvline(cut_to, color="red")
    plt.show()


def cut_chunk(signal, cut_from, cut_to, path_to_save):
    chunk=Signal.get_subsignal(signal, cut_from, cut_to)
    Signal.save_csv(chunk, path_to_save)
    restored=Signal.read_3d_csv(path_to_save)
    display_signal(restored)

parser=ArgumentParser()
parser.add_argument("--path", required=True)
parser.add_argument("--action", required=True,
                    choices=["show_all", "show_range", "cut"])
parser.add_argument("--cut_from", required=False, type=float)
parser.add_argument("--cut_to", required=False, type=float)
parser.add_argument("--output_file", required=False)
args=parser.parse_args()

signal=Signal.read_3d_csv(args.path)
if args.action=="show_all":
    display_signal(signal, display_now=True)
elif args.action=="show_range":
    show_chunks(signal, args.cut_from, args.cut_to)
elif args.action=="cut":
    cut_chunk(signal, args.cut_from, args.cut_to, args.output_file)
