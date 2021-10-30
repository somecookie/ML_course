# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np
import costs

def least_squares(y, tx):
    """calculate the least squares solution."""
    w = np.linalg.solve(tx.T@tx, tx.T@y)
    return costs.compute_mse(y, tx, w), w
