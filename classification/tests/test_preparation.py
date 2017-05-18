#! /usr/bin/python3
import unittest
import pandas as pd
import numpy as np
from numpy.testing import *
from classification.preparation import InvalidLogException, get_chunks
from pandas.util.testing import *


class SplitTestSuite(unittest.TestCase):
    
    def test_empty_log_split_failure(self):
        frame=pd.Series([], index=[]).to_frame()
        with self.assertRaises(InvalidLogException):
            get_chunks(frame, pd.to_timedelta("1s"))

    def test_unsufficient_chunk(self):
        timestamps=pd.date_range(pd.datetime.today(),
                                 freq="100ms", 
                                 periods=5).tolist()
        frame=pd.Series(range(0, 5), index=timestamps).to_frame()
        chunks=get_chunks(frame, pd.to_timedelta("1s"))
        self.assertEqual(0, len(chunks))

    def test_exact_single_chunk(self):
        timestamps=pd.date_range(pd.datetime.today(),
                                 freq="100ms", 
                                 periods=10).tolist()
        frame=pd.Series(range(0, 10), index=timestamps).to_frame()
        chunks=get_chunks(frame, pd.to_timedelta("1s"))
        expected_chunk=frame
        self.assertEqual(1, len(chunks))
        assert_frame_equal(expected_chunk, chunks[0])

    def test_stripped_single_chunk(self):
        timestamps=pd.date_range(pd.datetime.today(),
                                 freq="100ms", 
                                 periods=12).tolist()
        frame=pd.Series(range(0, 12), index=timestamps).to_frame()
        chunks=get_chunks(frame, pd.to_timedelta("1s"))
        expected_chunk=frame[:10]
        self.assertEqual(1, len(chunks))
        assert_frame_equal(expected_chunk, chunks[0])

    def test_exact_multiple_chunks(self):
        timestamps=pd.date_range(pd.datetime.today(),
                                 freq="100ms", 
                                 periods=30).tolist()
        frame=pd.Series(range(0, 30), index=timestamps).to_frame()
        chunks=get_chunks(frame, pd.to_timedelta("1s"))
        expected_chunks=frame[:10], frame[10:20], frame[20:]
        self.assertEqual(3, len(chunks))
        assert_frame_equal(expected_chunks[0], chunks[0])
        assert_frame_equal(expected_chunks[1], chunks[1])
        assert_frame_equal(expected_chunks[2], chunks[2])

    def test_excessive_multiple_chunks(self):
        timestamps=pd.date_range(pd.datetime.today(),
                                 freq="100ms", 
                                 periods=35).tolist()
        frame=pd.Series(range(0, 35), index=timestamps).to_frame()
        chunks=get_chunks(frame, pd.to_timedelta("1s"))
        expected_chunks=frame[:10], frame[10:20], frame[20:30]
        self.assertEqual(3, len(chunks))
        assert_frame_equal(expected_chunks[0], chunks[0])
        assert_frame_equal(expected_chunks[1], chunks[1])
        assert_frame_equal(expected_chunks[2], chunks[2])

