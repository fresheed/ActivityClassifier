#! /usr/bin/python3
import unittest
import numpy as np
from numpy.testing import *
from classification import dtw


class DTWTestSuite(unittest.TestCase):
    
    def test_evaluation_for_zero_signals(self):
        signal=np.array([0, 0, 0])
        etalon=np.array([0, 0, 0])
        self.assertEqual(0, dtw.evaluate_optimal_transform(signal, etalon))

    def test_evaluation_for_constants(self):
        signal=np.array([1, 1, 1])
        etalon=np.array([2, 2, 2])
        self.assertEqual(3, dtw.evaluate_optimal_transform(signal, etalon))

    def test_evaluation_for_shifted_signal(self):
        signal=np.array([1, 3, 5])
        etalon=np.array([-3, -1, 1])
        self.assertEqual(12, dtw.evaluate_optimal_transform(signal, etalon))

    def test_evaluation_for_warped_signal(self):
        signal=np.array([1, 2, 3])
        etalon=np.array([2, 4, 6])
        self.assertEqual(5, dtw.evaluate_optimal_transform(signal, etalon))

    def test_different_lengths_signals(self):
        signal=np.array([1, 2])
        etalon=np.array([3, 5, 6, 7])
        self.assertEqual(14, dtw.evaluate_optimal_transform(signal, etalon))
