# -*- coding: utf-8 -*-
import numpy as np
"""Function used to compute the loss."""

def compute_mse(y, tx, w):
    """
    Calculate the mse loss.
    """
    e = y - np.matmul(tx, w)
    N = len(y)
    return 1/(2*N)*(e.T@e)

def compute_rmse(y, tx, w):
    """Calculate the loss.
    """

    return np.sqrt(2*compute_mse(y, tx, w))

def compute_mae(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - np.matmul(tx, w)
    N = len(y)
    return np.mean(e)/N