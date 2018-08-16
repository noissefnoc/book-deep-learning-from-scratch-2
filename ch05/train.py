#!/usr/bin/env python

from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from dataset import ptb
from ch05.simple_rnnlm import SimpleRnnlm


if __name__ == '__main__':
    # hyper parameters setting
    batch_size = 10
    wordvec_size = 100
    hidden_size = 100 # number of RNN hidden state vector elements
    time_size = 5     # size of RNN expansion
    lr = 0.1
    max_epoch = 100

    # training data loading
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    corpus_size = 1000 # test dataset size
    corpus = corpus[:corpus_size]
    vocab_size = int(max(corpus) + 1)
    xs = corpus[:-1] # input
    ts = corpus[1:]  # otuput (teaching label)

    # model creation
    model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
    optimizer = SGD(lr)
    trainer = RnnlmTrainer(model, optimizer)

    trainer.fit(xs, ts, max_epoch, batch_size, time_size)
    trainer.plot()
