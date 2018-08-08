#!/usr/bin/env python

import numpy as np
from common.optimizer import SGD
from dataset import spiral
import matplotlib.pyplot as plt
from ch01.two_layer_net import TwoLayerNet


if __name__ == '__main__':
    # 1. hyper parameter settings
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    learning_rate = 1.0

    # 2. load data and generate model and optimizer
    x, t = spiral.load_data()
    model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
    optimizer =SGD(lr=learning_rate)

    # variables for learning
    data_size = len(x)
    max_iters = data_size // batch_size
    total_loss = 0
    loss_count = 0
    loss_list = []

    for epoch in range(max_epoch):
        # 3. shuffle data
        idx = np.random.permutation(data_size)
        x = x[idx]
        t = t[idx]

        for iters in range(max_iters):
            batch_x = x[iters * batch_size: (iters + 1) * batch_size]
            batch_t = t[iters * batch_size: (iters + 1) * batch_size]

            # 4. process grads and update parameters
            loss = model.forward(batch_x, batch_t)
            model.backward()
            optimizer.update(model.params, model.grads)

            total_loss += loss
            loss_count += 1

            # 5. output learning result
            if (iters + 1) % 10 == 0:
                avg_loss = total_loss / loss_count
                print('| epoch %d | iter %d / %d | loss %.2f'
                      % (epoch + 1, iters + 1, max_iters, avg_loss))
                loss_list.append(avg_loss)
                total_loss, loss_count = 0, 0
