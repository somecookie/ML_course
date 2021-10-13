# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient computation.It's same as the gradient descent.
    # ***************************************************
    N = len(y)
    e = y - np.matmul(tx, w)
    return -(1/N)*tx.T@e


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient descent.
    # ***************************************************
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for b_y, b_tx in batch_iter(y, tx, batch_size):
            loss = losses.append(compute_loss(y, tx, w))
            w = w - gamma * compute_stoch_gradient(b_y, b_tx, w)
            ws.append(w)
    return losses, ws