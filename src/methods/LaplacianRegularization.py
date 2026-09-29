import torch


def _nabla(matrix,J,lambda_,Y,L):

    return J*matrix - Y + lambda_* L @ matrix

def _matrix_product(matrix1,matrix2):
    # per-column (per-time-step) inner product; columns are independent problems
    return (matrix1*matrix2).sum(dim=0, keepdim=True)


def LR(Y,J,L,lambda_ = 10**0,K = 50):
    """Laplacian regularization (Smola and Kondor, Learning Theory and Kernel Machines, 2003),
    solved by the conjugate gradient method.

    Each time step (column) is restored independently (Laplacian regularization has no temporal
    coupling); all columns are solved at once, with a per-column step size.

    Parameters
    ----------
    Y : Array(NxM)
        sampled data
    J : Array(NxM)
        sampling operator
        each element must have 0 or 1
    L : Array(NxN)
        graph laplacian matrix
    lambda_ : int
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

    X = torch.zeros_like(Y)
    delta_X = -1*_nabla(X,J,lambda_,Y,L)

    for _ in range(K):

        # Stepsize decision (per column)
        nablax = _nabla(X,J,lambda_,Y,L)
        tau = -1 * _matrix_product(delta_X,nablax)/(_matrix_product(delta_X,_nabla(delta_X,J,lambda_,Y,L)+Y)+MIN_NUM)
        # Search direction updating
        X_next = X + tau*delta_X
        nablaXnext = _nabla(X_next,J,lambda_,Y,L)
        gamma = _matrix_product(nablaXnext,nablaXnext)/(_matrix_product(nablax,nablax)+MIN_NUM)
        delta_X = -1* nablaXnext + gamma * delta_X
        X = X_next

    return X.cpu().numpy()
