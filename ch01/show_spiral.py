#!/usr/bin/env python

from dataset import spiral
import matplotlib.pyplot as plt


if __name__ == '__main__':
    x, t = spiral.load_data()
    print('x', x.shape)  # (300, 2)
    print('t', t.shape)  # (300, 3)

    #
    N = 100
    CLS_NUM = 3
    markers = ['o', 'x', '^']

    for i in range(CLS_NUM):
        plt.scatter(x[i * N:(i + 1) * N, 0], x[i * N:(i + 1) * N, 1], s=40, marker=markers[i])

    plt.show()