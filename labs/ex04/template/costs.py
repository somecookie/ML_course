# -*- coding: utf-8 -*-
import numpy as np
"""Function used to compute the loss."""

def compute_mse_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - np.matmul(tx, w)
    N = len(y)
    return 1/(2*N)*np.dot(e, e)

def compute_mae_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - np.matmul(tx, w)
    N = len(y)
    return np.mean(e)/N