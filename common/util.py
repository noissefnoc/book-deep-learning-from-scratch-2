#!/usr/bin/env python

import numpy as np


def clip_grads(grads, max_norm):
    total_norm = 0

    for grad in grads:
        total_norm += np.sum(grad ** 2)

    total_norm = np.sqrt(total_norm)
    rate = max_norm / (total_norm + 1e-6)

    if rate < 1:
        for grad in grads:
            grad *= rate


# 2.3.1 Prepossessing by Python
def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .') # split comma as a word
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word
