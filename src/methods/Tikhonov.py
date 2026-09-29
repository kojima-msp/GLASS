import torch


def _nabla(matrix,J,D_h,beta,gamma,Y,L):

    return J*matrix - Y + beta*L@matrix + gamma*matrix@D_h@D_h.T

def _matrix_product(matrix1,matrix2):
    return (matrix1*matrix2).sum()


def Tikhonov(Y,J,L,K = 50,gamma = 1.0,beta = 1.0):
    """
    Parameters
    ----------
    Y : Array(NxM)
        sampled data
    J : Array(NxM)
        sampling operator
        each element must have 0 or 1
    L : Array(NxN)
        graph laplacian matrix
    K : int
        maximum number of iterations
    gamma : float
        regularization parameter of the temporal difference
    beta : float
        regularization parameter of the graph Laplacian

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
    delta_X = -1*_nabla(X,J,D_h,beta,gamma,Y,L)

    for _ in range(K):

        # Stepsize decision
        nablax = _nabla(X,J,D_h,beta,gamma,Y,L)
        tau = -1 * _matrix_product(delta_X,nablax)/(_matrix_product(delta_X,_nabla(delta_X,J,D_h,beta,gamma,Y,L)+Y)+MIN_NUM)
        # Search direction updating
        X_next = X + tau*delta_X
        nablaXnext = _nabla(X_next,J,D_h,beta,gamma,Y,L)
        gamma_cg = _matrix_product(nablaXnext,nablaXnext)/(_matrix_product(nablax,nablax)+MIN_NUM)
        delta_X = -1* nablaXnext + gamma_cg * delta_X
        X = X_next

    return X.cpu().numpy()
