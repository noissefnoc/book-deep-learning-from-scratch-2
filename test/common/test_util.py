#!/usr/bin/env python

import unittest
import numpy as np
import numpy.testing as npt
from common.util import preprocess, create_co_matrix, cos_similarity


class TestUtil(unittest.TestCase):
    # 2.3.1 Prepossessing by Python
    def test_preprocess(self):
        input = 'You say goodbye and I say hello.'
        expected_word_to_id = {
            "you": 0,
            "say": 1,
            "goodbye": 2,
            "and": 3,
            "i": 4,
            "hello": 5,
            ".": 6
        }
        expected_id_to_word = {
            0: "you",
            1: "say",
            2: "goodbye",
            3: "and",
            4: "i",
            5: "hello",
            6: "."
        }
        expected_corpus = np.array([0, 1, 2, 3, 4, 1, 5, 6])

        actual_corpus, actual_word_to_id, actual_id_to_word = preprocess(input)

        npt.assert_array_equal(
            actual_corpus,
            expected_corpus)

        self.assertDictEqual(
            actual_word_to_id,
            expected_word_to_id)

        self.assertDictEqual(
            actual_id_to_word,
            expected_id_to_word)

    # 2.3.4 cooccurrence matrix
    def test_create_co_matrix(self):
        corpus, _, _ = preprocess('You say goodbye and I say hello.')
        vocab_size = np.unique(corpus).size

        expected = np.array([
           [0, 1, 0, 0, 0, 0, 0],
           [1, 0, 1, 0, 1, 1, 0],
           [0, 1, 0, 1, 0, 0, 0],
           [0, 0, 1, 0, 1, 0, 0],
           [0, 1, 0, 1, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 1, 0],
        ], dtype=np.int32)

        actual = create_co_matrix(corpus, vocab_size)

        npt.assert_array_equal(actual, expected)

    # 2.3.5 vector similarity
    def test_cos_similarity(self):
        input_x = [np.array([0, 0]), np.array([3, 3, 3, 3])]
        input_y = [np.array([0, 0]), np.array([4, 4, 4, 4])]
        expected = [0, 1]

        for i in range(len(input_x)):
            actual = cos_similarity(input_x[i], input_y[i])
            self.assertAlmostEqual(actual, expected[i])
