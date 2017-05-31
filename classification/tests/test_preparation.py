#! /usr/bin/python3
import unittest
import pandas as pd
import numpy as np
from numpy.testing import *
from classification.preparation import InvalidLogException, get_chunks, downsample
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


class DownsamplingTestSuite(unittest.TestCase):

    def to_timestamps(self, *mss):
        def to_timestamp(total_milliseconds):
            seconds=total_milliseconds // 1000
            left_milliseconds=total_milliseconds % 1000
            left_microseconds=left_milliseconds * 1000
            to_parse="%d/%06d" % (seconds, left_microseconds)
            timestamp=pd.to_datetime(to_parse, format="%S/%f")
            return timestamp

        return [to_timestamp(ms) for ms in mss]

    def get_frame(self, values, mss):
        timestamps=self.to_timestamps(*mss)
        frame=pd.Series(values, index=timestamps).to_frame()
        return frame

    def test_single_value_downsampled(self):
        ts=self.to_timestamps(1200)
        frame=pd.Series([1], index=ts).to_frame()
        downsampled=downsample(frame, 100)
        assert_frame_equal(frame, downsampled)

    def test_evens_downsampled(self):
        frame=self.get_frame([1, 2, 3, 1, 2, 3], [0, 33, 66, 100, 133, 166])
        downsampled=downsample(frame, 100)
        new_index=self.to_timestamps(0, 100)
        expected=pd.Series([2, 2], new_index).to_frame()
        assert_frame_equal(expected, downsampled)

    def test_unevens_downsampled(self):
        ts=self.to_timestamps(0, 10, 11, 50, 99, 100, 150, )
        frame=pd.Series([1, 2, 3, 4, 5, 1, 2], index=[ts]).to_frame()
        frame=self.get_frame([1, 2, 3, 4, 5, 1, 2],
                             [0, 10, 11, 50, 99, 100, 150,])
        downsampled=downsample(frame, 100)
        new_index=self.to_timestamps(0, 100)
        expected=pd.Series([3, 1.5], new_index).to_frame()
        assert_frame_equal(expected, downsampled)

    
