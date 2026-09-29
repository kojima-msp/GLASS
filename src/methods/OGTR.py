import numpy as np
import torch


def _nabla(matrix,J,D_h,lambda_,Y,L):

    return J*matrix - Y + lambda_* L @ matrix @ D_h @ D_h.T

def _matrix_product(matrix1,matrix2):
    return (matrix1*matrix2).sum()


def OGTR(Y,J,L,lambda_ = 10**0,K = 50):
    """Time-varying graph signal reconstruction (Qiu et al., IEEE Journal of Selected Topics in
    Signal Processing, 2017), solved by the conjugate gradient method.

    paper : https://ieeexplore.ieee.org/document/7979523

    Parameters
    ----------
    Y : Array(NxM)
        sampled data
    J : Array(NxM)
        sampling operator
        each element must have 0 or 1
    L : Array(NxN)
        graph laplacian matrix
    lambda_ : float
        regularization parameter
    K : int
        maximum number of iterations

    Returns
    ----------
    X : Array(NxM)
        reconstructed signal
    """

    Y = torch.tensor(Y)
    J = torch.tensor(J)
    L = torch.tensor(L)

    MIN_NUM = 10**-10

    # initialization
    M = Y.shape[1]
    D_h = (-1*torch.eye(M) + torch.diag(torch.ones(M-1), -1))[:, :M-1]

    X = torch.zeros_like(Y)
    delta_X = -1*_nabla(X,J,D_h,lambda_,Y,L)

    for _ in range(K):

        # Stepsize decision
        nablax = _nabla(X,J,D_h,lambda_,Y,L)
        tau = -1 * _matrix_product(delta_X,nablax)/(_matrix_product(delta_X,_nabla(delta_X,J,D_h,lambda_,Y,L)+Y)+MIN_NUM)
        # Search direction updating
        X_next = X + tau*delta_X
        nablaXnext = _nabla(X_next,J,D_h,lambda_,Y,L)
        gamma = _matrix_product(nablaXnext,nablaXnext)/(_matrix_product(nablax,nablax)+MIN_NUM)
        delta_X = -1* nablaXnext + gamma * delta_X
        X = X_next

    return X.cpu().numpy()


def OGTR_online(Y,J,L,lambda_ = 10**0,K = 50):
    """Online OGTR (Qiu et al., JSTSP 2017, Sec. V) built on the batch solver.

    The online problem at time t is the batch problem on the two-column window
    [x_hat_{t-1}, current]: for M=2 the temporal difference D_h reduces to [-1, 1]^T, so
    tr((XD_h)^T L XD_h) is exactly the online regularizer. The previous estimate enters as a
    fully observed prior, and the first frame gets an unobserved zero instead (J=0).

    Parameters mirror OGTR; returns X (NxM).
    """
    Y = np.asarray(Y)
    J = np.asarray(J)
    N, M = Y.shape
    dtype = Y.dtype  # keep the data dtype so OGTR's internal tensors stay consistent
    X = np.zeros((N, M), dtype=dtype)

    ones = np.ones(N, dtype=dtype)
    zero = np.zeros(N, dtype=dtype)
    for t in range(M):
        prev, prev_j = (X[:, t-1], ones) if t > 0 else (zero, zero)
        Y2 = np.stack([prev, Y[:, t]], axis=1)
        J2 = np.stack([prev_j, J[:, t].astype(dtype)], axis=1)
        X[:, t] = OGTR(Y2, J2, L, lambda_=lambda_, K=K)[:, -1]

    return X
