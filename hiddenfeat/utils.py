import numpy as np
import random
import scipy

def inverse_norm(b, A):
    """Norm weighted by matrix inverse: \|b\|_{A^{-1}}"""
    return np.dot(b, np.linalg.solve(A,b))

def seed_all(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    
def feature_basis(features):
    assert len(np.shape(features)) == 3
    n = features.shape[0]
    m = features.shape[1]
    d = features.shape[2]
    
    matrix = np.reshape(features, (n*m, d))
    orth = scipy.linalg.orth(matrix)
    return np.reshape(orth, (n , m, orth.shape[-1]))

def get_orthogonal(basis):
    assert len(np.shape(basis)) == 3
    n = np.shape(basis)[0]
    m = np.shape(basis)[1]
    d = np.shape(basis)[2]
    assert d < n*m
    
    matrix = np.reshape(basis, (n*m, d))
    newcol = np.random.rand(n*m, 1)
    A = np.hstack((matrix, newcol))
    b = np.zeros(d + 1)
    b[-1] = 1
    x = np.linalg.lstsq(A.T, b, rcond=None)[0]
    return np.reshape(x, (n, m, 1))

def basis_completion(basis, fulldim=None):
    n = np.shape(basis)[0]
    m = np.shape(basis)[1]
    d = np.shape(basis)[2]
    matrix = np.reshape(basis, (n*m, d))
    assert d <= n*m and np.linalg.matrix_rank(matrix) == d
    
    if fulldim is None and d==n*m or d==fulldim:
        return basis
    
    if fulldim is None or fulldim==n*m:
        X = scipy.linalg.null_space(matrix.T)
        return np.reshape(X, (n, m, n*m - d))
    else:
        assert fulldim < n*m
        while basis.shape[-1] < fulldim:
            x = get_orthogonal(basis)
            basis = np.dstack((basis, x))
        return basis[:, :,-(fulldim - d):]