#!/usr/bin/env python

import numpy as np
from common.layers import MatMul, SoftmaxWithLoss


class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size

        # initialize weights
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        # create layer
        self.in_layer_0 = MatMul(W_in)
        self.in_layer_1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        # combine all weights and grads into list
        layers = [self.in_layer_0, self.in_layer_1, self.out_layer]
        self.params, self.grads = [], []

        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # set word vector into member variable
        self.word_vecs = W_in

    def foward(self, contexts, target):
        h0 = self.in_layer_0.forward(contexts[:, 0])
        h1 = self.in_layer_1.forward(contexts[:, 1])
        h = (h0 + h1) * 0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer_1.backward(da)
        self.in_layer_0.backward(da)
        return None
