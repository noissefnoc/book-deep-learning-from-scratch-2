#!/usr/bin/env python

import unittest
import numpy as np
import numpy.testing as npt
from ch04.negative_sampling_layer import EmbeddingDot


class TestEmbeddingDot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        W = np.array([
            [ 0,  1,  2],
            [ 3,  4,  5],
            [ 6,  7,  8],
            [ 9, 10, 11],
            [12, 13, 14],
            [15, 16, 17],
            [18, 19, 20]])
        cls.EmbeddingDot = EmbeddingDot(W)

    def test_forward(self):
        h = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]])
        idx = np.array([0, 3, 1])
        expected = np.array([5, 122, 86])

        actual = self.EmbeddingDot.forward(h, idx)

        npt.assert_array_equal(expected, actual)

    def test_backward(self):
        h = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]])
        idx = np.array([0, 3, 1])
        expected = np.array([
            [0, 5, 10],
            [1098, 1220, 1342],
            [258, 344, 430]])

        dout = self.EmbeddingDot.forward(h, idx)
        actual = self.EmbeddingDot.backward(dout)

        npt.assert_array_equal(expected, actual)
