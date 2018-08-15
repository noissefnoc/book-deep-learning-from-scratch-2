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


# 2.3.6 ranking of similar word
def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    """
    ranking of similar word
    :param query: query word
    :param word_to_id: dictionary of word to id
    :param id_to_word: dictionary of id to word
    :param word_matrix: matrix of combining word vector
    :param top: number of ranking
    :return: None
    """
    # 1. get query
    if query not in word_to_id:
        print('%s is not found' % query)
        return

    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # 2. calculate cosine similarity
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)

    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # 3. print top result of cosine similarity
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue

        print('%s: %s' % (id_to_word[i], similarity[i]))

        count += 1

        if count >= top:
            return


# 2.4.1 Pointwise Mutual Information (PMI)
def ppmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j] * S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total // 100) == 0:
                    print('%.1f%% done' % (100 * cnt / total))

    return M


# 3.3.1 contexts and target
def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size:-window_size]
    contexts = []

    for idx in range(window_size, len(corpus) - window_size):
        cs = []

        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)

    return np.array(contexts), np.array(target)


# 3.3.2 convert to one-hot vector
def convert_one_hot(corpus, vocab_size):
    """
    convert to one hot vector
    :param corpus: word id list (1 dimension or 2 dimension NumPy array)
    :param vocab_size: vocabulary size
    :return: one-hot vector (2 dimension or 3 dimension NumPy array)
    """
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)

        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot


# 4.3.2 CBOW model training code
def to_cpu(x):
    import numpy
    if type(x) == numpy.ndarray:
        return x

    return np.asnumpy(x)
