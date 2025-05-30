# Necessary packages
import numpy as np
import pandas as pd

# PyMC3 for Bayesian Inference
# import pymc3 as pm
from sklearn.preprocessing import StandardScaler

def binary_sampler(p, rows, cols):
    '''Sample binary random variables.

    Args:
    - p: probability of 1
    - rows: the number of rows
    - cols: the number of columns

    Returns:
    - binary_random_matrix: generated binary random matrix.
    '''
    unif_random_matrix = np.random.uniform(0., 1., size=[rows, cols])
    binary_random_matrix = 1 * (unif_random_matrix < p)
    return binary_random_matrix.astype('float32')


def sample_batch_index(total, batch_size):
    '''Sample index of the mini-batch.

    Args:
    - total: total number of samples
    - batch_size: batch size

    Returns:
    - batch_idx: batch index
    '''
    total_idx = np.random.permutation(total)
    batch_idx = total_idx[:batch_size]
    return batch_idx