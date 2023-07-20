"""Auxiliary methods for kalman filter update."""
import numpy as np
import pandas as pd
import copy
from scipy import stats
from packages.dlmtf.src.utils import tidy_parameters
from scipy.linalg import block_diag


def _build_W(mod, P: np.array):
    p = P.shape[0]
    discount_matrix = np.ones([p, p])
    np.fill_diagonal(discount_matrix, 1 / mod.discount_factors)

    W = P * discount_matrix - P

    return W


def _calc_predictive_mean_and_var(F: np.array, a: np.array,
                                  R: np.array, s: float):
    f = F.T @ a
    q = F.T @ R @ F + s
    return np.ravel(f), np.ravel(q)


def _calc_aR(mod):
    m = mod.m
    C = mod.C
    G = mod.G
    discount_matrix = mod.discount_matrix

    a = G @ m
    P = G @ C @ G.T
    W = _build_W(P=P, discount_matrix=discount_matrix)
    R = P + W

    return a, R


def _build_Gnonlinear(m: np.array, order: int):
    response_block_index = np.arange(0, order)
    decay_block_index = np.arange(order, 2 * order)

    diag_order = np.identity(order)

    response_block = m[response_block_index, 0]
    decay_block = m[decay_block_index, 0]

    diag_decay_block = np.identity(order)[:order-1, :]
    diag_response_block = np.diag(response_block)[1:, :] * 0

    nonlinear_block = np.block([[decay_block, response_block],
                               [diag_decay_block, diag_response_block],
                               [0 * diag_order, diag_order]])
    nonlinear_block = nonlinear_block.reshape(2 * order, 2 * order)

    assert len(decay_block) == len(response_block), \
        'decay and responde blocks differs!'

    assert len(decay_block) == len(response_block), \
        'decay and responde blocks differs!'

    return nonlinear_block
