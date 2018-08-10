#!/usr/bin/env python

import numpy as np
import unittest
import numpy.testing as npt
from common.functions import sigmoid, relu, softmax, cross_entropy_error


# TODO: create helper function for easing table driven test
class TestFunctions(unittest.TestCase):
    def test_sigmoid_scalar(self):
        inputs = [-np.inf, 0, np.inf]
        expected = [0, 0.5, 1]

        for i in range(len(inputs)):
            actual = sigmoid(inputs[i])
            self.assertEqual(
                actual,
                expected[i],
                "expected=[%d] but got actual=[%d]" % (expected[i], actual))

    def test_relu_scalar(self):
        inputs = [-1, 0, 1]
        expected = [0, 0, 1]

        for i in range(len(inputs)):
            actual = relu(inputs[i])
            self.assertEqual(
                actual,
                expected[i],
                "expected=[%d] but got actual=[%d]" % (expected[i], actual))

    def test_relu_vector(self):
        inputs = [np.array([-1, 0, 1])]
        expected = [np.array([0, 0, 1])]

        for i in range(len(inputs)):
            actual = relu(inputs[i])
            npt.assert_array_equal(
                actual,
                expected[i])

    def test_softmax(self):
        inputs = [np.array([0])]
        expected = [np.array([1])]

        for i in range(len(inputs)):
            actual = softmax(inputs[i])
            npt.assert_array_equal(
                actual,
                expected[i])


if __name__ == '__main__':
    unittest.main()
