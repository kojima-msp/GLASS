import torch


def _nabla(matrix,J,D_h,lambda_,Y,L_beta):

    return J*matrix - Y + lambda_* L_beta @ matrix @ D_h @ D_h.T

def _matrix_product(matrix1,matrix2):
    return (matrix1*matrix2).sum()

def _fractional_power(A,beta):
    # A is real symmetric (graph Laplacian + epsilon*I, PSD) -> exact via eigendecomposition
    evals, evecs = torch.linalg.eigh(A)
    return (evecs * evals.clamp(min=0)**beta) @ evecs.T

def TRSS(Y,J,L,lambda_ = 10**0,epsilon = 0.5,beta = 1.5,K = 50):
    """Reconstruction of time-varying graph signals via Sobolev smoothness
    (Giraldo et al., IEEE Transactions on Signal and Information Processing over Networks, 2022),
    solved by the conjugate gradient method.

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
    epsilon : float
        shift of the Sobolev operator (L + epsilon I)^beta
    beta : float
        exponent of the Sobolev operator
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

    # (L + epsilon I)^beta is constant over iterations -> precompute once
    L_beta = _fractional_power(L + epsilon*torch.eye(L.shape[0]), beta)

    X = torch.zeros_like(Y)
    delta_X = -1*_nabla(X,J,D_h,lambda_,Y,L_beta)

    for _ in range(K):

        # Stepsize decision
        nablax = _nabla(X,J,D_h,lambda_,Y,L_beta)
        tau = -1 * _matrix_product(delta_X,nablax)/(_matrix_product(delta_X,_nabla(delta_X,J,D_h,lambda_,Y,L_beta)+Y)+MIN_NUM)
        # Search direction updating
        X_next = X + tau*delta_X
        nablaXnext = _nabla(X_next,J,D_h,lambda_,Y,L_beta)
        gamma = _matrix_product(nablaXnext,nablaXnext)/(_matrix_product(nablax,nablax)+MIN_NUM)
        delta_X = -1* nablaXnext + gamma * delta_X
        X = X_next

    return X.cpu().numpy()
