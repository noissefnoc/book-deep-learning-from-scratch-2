#!/usr/bin/env python

import unittest
import numpy as np
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


if __name__ == '__main__':
    unittest.main()