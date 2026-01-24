import numpy as np
import torch
from scipy.stats import *
import itertools
import scipy
from scipy.special import logsumexp

import sys


def sum_along_axis(a, axis=1):
    # assumes it is a numpy array
    if a.ndim <= axis:
        return a
    else:
        return a.sum(axis=axis)


def make_pos_def_torch(x):
    """
    Force a square symmetric matrix to be positive semi definite
    """
    w, v = torch.linalg.eigh(torch.real(x))
    w_pos = torch.clip(w, 0, None)
    nonzero_w = w_pos[w_pos > 0]
    w_new = w_pos
    if nonzero_w.shape[0] > 0:
        if nonzero_w.shape[0] < w.shape[0]:
            # min_w = max(torch.max(nonzero_w)*1e-5,torch.min(nonzero_w))
            min_w = max(torch.max(nonzero_w) * 0.1, torch.min(nonzero_w))
            w_new = w_pos + min_w
    elif nonzero_w.shape[0] == 0:
        eprint(
            "No positive eigenvalues for A {}. w={} {}, w_pos={} {}, nonzero_w={}".format(
                x, w, w.shape, w_pos, w_pos.shape, nonzero_w
            )
        )
        w_new = torch.ones_like(w)
    x_star = v @ np.diag(w_new) @ v.T
    p = torch.sqrt(torch.sum(torch.abs(w)) / torch.sum(torch.abs(w_new)))
    return p * x_star


def make_pos_def(x):
    """
    Force a square symmetric matrix to be positive semi definite
    """
    w, v = np.linalg.eigh(x)
    w_pos = np.clip(w, 0, None)
    nonzero_w = w_pos[w_pos > 0]
    w_new = w_pos
    if nonzero_w.shape[0] > 0:
        if nonzero_w.shape[0] < w.shape[0]:
            min_w = max(np.max(nonzero_w) * 1e-5, np.min(nonzero_w))
            w_new = w_pos + min_w
    else:
        eprint(
            "No positive eigenvalues for A {}. w={} {}, w_pos={} {}, nonzero_w={}".format(
                x, w, w.shape, w_pos, w_pos.shape, nonzero_w
            )
        )
        w_new = np.ones_like(w)
    # w_neg = np.abs(np.clip(w,None,0))
    x_star = v @ np.diag(w_new) @ v.T
    p = np.sqrt(np.sum(np.abs(w)) / np.sum(w_new))
    return p * x_star


def safe_cholesky(A: torch.Tensor) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    eigenvalues = torch.clamp(eigenvalues, min=1e-10)  # set any negative eigenvalues to a small positive value
    # L = eigenvectors @ torch.diag(eigenvalues.sqrt()) @ eigenvectors.t() #ZCA
    __, Rqr = torch.linalg.qr(torch.diag(eigenvalues.sqrt()) @ eigenvectors.t())
    Dg = torch.diag(torch.sign(torch.diag(Rqr)))
    Rqr = Dg @ Rqr
    return Rqr.T
