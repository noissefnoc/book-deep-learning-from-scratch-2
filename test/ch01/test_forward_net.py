#!/usr/bin/env python

import unittest
import numpy as np
import numpy.testing as npt
from ch01.forward_net import Sigmoid, Affine, TwoLayerNet


class TestSigmoid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sigmoid = Sigmoid()

    def test_forward_scalar(self):
        input = [-np.inf, 0, np.inf]
        expected = [0, 0.5, 1]

        for i in range(len(input)):
            actual = self.sigmoid.forward(input[i])
            self.assertEqual(actual, expected[i])


class TestAffine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        W = np.array([[1, 0],
                     [0, 1]])
        b = np.array([1, 0])
        cls.affine = Affine(W, b)

    def test_forward(self):
        x = np.array([1, 1])
        expected = np.array([2, 1])
        actual = self.affine.forward(x)
        npt.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()