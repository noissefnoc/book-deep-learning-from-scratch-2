#!/usr/bin/env python

from common.util import preprocess, create_co_matrix, cos_similarity


if __name__ == '__main__':
    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)
    C = create_co_matrix(corpus, vocab_size)

    c0 = C[word_to_id['you']] # word vector of 'you'
    c1 = C[word_to_id['i']]   # word vector of 'i'
    print(cos_similarity(c0, c1))
