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


# 2.3.4 cooccurrence matrix
def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


# 2.3.5 vector similarity
def cos_similarity(x, y, eps=1e-8):
    # adds tiny value for preventing divide by zero
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)