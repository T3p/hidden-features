import numpy as np
import random

def inverse_norm(b, A):
    """Norm weighted by matrix inverse: \|b\|_{A^{-1}}"""
    return np.dot(b, np.linalg.solve(A,b))

def seed_all(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)