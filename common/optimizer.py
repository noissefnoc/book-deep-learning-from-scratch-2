#!/usr/bin/env python

import numpy as np


# 1.3.6 Update weight
# Stochastic Gradient Descent
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
