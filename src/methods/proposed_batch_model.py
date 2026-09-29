import random

import torch
from torch import nn
from torch.nn import functional as F


class Batch_Model(nn.Module):
    """Time-varying Graph Signal Batch Reconstruction Model using unrolling (TRSS-based).

    Unrolls the TRSS (Sobolev-smoothness) reconstruction: the regularizer uses the Sobolev
    operator (L + eps I)^beta instead of the plain Laplacian L, and the temporal difference is a
    learnable multi-tap FIR filter. lambda, eps, beta and the FIR taps are all trainable.
    """
    def __init__(self,layer:int,lambda_init: float | None=0.0,n_taps: int= 5,
                 eps_init: float | None=1.0,beta_init: float | None=1.0):
        """__init__
        Parameters
        ----------
        layer : int
            depth of layers
        lambda_init : float
            init lambda (default:0.0)
        n_taps : int
            number of learnable FIR taps (L in the paper, default:5)
        eps_init, beta_init : float
            initial eps / beta of the Sobolev operator (L + eps I)^beta. eps > 0 removes L's null
            space, which plain L cannot regularize at all. 
        """
        super().__init__()
        self.n_taps = n_taps

        # trainable params, one per layer
        # lambda_:1 x len(layer)
        self.lambda_ = nn.ParameterList([nn.Parameter(torch.tensor([lambda_init])) for _ in range(layer)])

        # Sobolev operator (L + eps I)^beta, one eps/beta per layer (stored as log for eps,beta > 0)
        self.eps_ = nn.ParameterList([nn.Parameter(torch.tensor([eps_init]).log()) for _ in range(layer)])
        self.beta_ = nn.ParameterList([nn.Parameter(torch.tensor([beta_init]).log()) for _ in range(layer)])

        # d_h:len(layer) x (self.n_taps)
        # the taps are softmax(d_h[1:]); random init -> near-uniform taps after the softmax.
        self.d_h = nn.ParameterList([nn.Parameter(torch.tensor([random.random() for _ in range(self.n_taps+1)]))
                                     for __ in range(layer)])

        self.layer = layer

    def _taps(self,z):
        """raw parameter (length n_taps+1) -> FIR taps d_1..d_L, non-negative and summing to 1.
        z[0] is unused: it keeps the tap index 1-origin, matching d_1..d_L in the paper."""
        return torch.softmax(z[1:self.n_taps+1], dim=0)

    def precompute_eig(self,L):
        """Eigendecompose L once (it is constant across layers). eps/beta of each layer then enter
        only through the eigenvalues, so the per-layer Sobolev operator is cheap and differentiable."""
        with torch.no_grad():
            self._evals, self._evecs = torch.linalg.eigh(L)
        self._evals = self._evals.clamp(min=0)

    def _sobolev(self,ite):
        """layer ite's Sobolev operator (L + eps_ite I)^beta_ite, built from the cached eigenpairs."""
        evals = self._evals + torch.exp(self.eps_[ite])
        return self._evecs @ torch.diag_embed(evals ** torch.exp(self.beta_[ite])) @ self._evecs.transpose(-1, -2)

    def _nabla(self,matrix,w,lambda_,Y,J,S):
        # matrix @ D_L == valid conv1d(matrix, w), and @ D_L.T its adjoint (conv_transpose1d).
        # S is the Sobolev operator of the layer.
        B, N, M = matrix.shape
        k = w.view(1, 1, -1)
        g = F.conv1d(matrix.reshape(B * N, 1, M), k).reshape(B, N, -1)
        diff = F.conv_transpose1d(g.reshape(B * N, 1, -1), k).reshape(B, N, M)
        return torch.mul(J, matrix) - Y + lambda_ * torch.einsum('bij,bjm->bim', S, diff)

    def _matrix_product(self,matrix1,matrix2):
        # per-sample Frobenius inner product -> (B, 1, 1): each sample gets its own CG step size.
        return torch.sum(torch.mul(matrix1, matrix2), dim=(1, 2), keepdim=True)

    def _extend_data(self,Y,J):
        # mirror-pad n_taps-1 frames so that X D_L is defined on the first frames; stripped in forward()
        mirror_Y = torch.flip(Y[:, :, 1:self.n_taps], dims=[2])
        mirror_J = torch.flip(J[:, :, 1:self.n_taps], dims=[2])
        return torch.cat([mirror_Y, Y], dim=2), torch.cat([mirror_J, J], dim=2)

    def forward(self,Y,J,L,precomputed_eig=False):
        """Batch reconstruction: unrolled conjugate gradient on the TRSS objective.

        Parameters
        ----------
        Y : Array(NxM) or Array(BxNxM)
            sampled data
        J : Array(NxM) or Array(BxNxM)
            sampling operator
            each element must have 0 or 1
        L : Array(NxN) or Array(BxNxN)
            graph laplacian matrix
            2-D L is shared across the batch; 3-D L is per-sample (syn uses a graph per realization)

        Returns
        ----------
        X : Array(NxM) or Array(BxNxM)
            reconstructed signal (same rank as Y)
        """
        dev = self.lambda_[0].device
        dt = torch.get_default_dtype()
        Y = torch.as_tensor(Y, dtype=dt, device=dev)
        J = torch.as_tensor(J, dtype=dt, device=dev)
        L = torch.as_tensor(L, dtype=dt, device=dev)

        squeeze = (Y.dim() == 2)
        if squeeze:
            Y = Y.unsqueeze(0); J = J.unsqueeze(0)
        B = Y.shape[0]
        if L.dim() == 2:
            L = L.unsqueeze(0).expand(B, -1, -1)

        # a window shorter than the FIR (online start-up) is padded up front with unobserved frames
        pad = max(self.n_taps + 1 - Y.shape[2], 0)
        if pad:
            z = torch.zeros(B, Y.shape[1], pad, dtype=dt, device=dev)
            Y = torch.cat([z, Y], dim=2)
            J = torch.cat([z, J], dim=2)

        Y, J = self._extend_data(Y,J)

        if not precomputed_eig:
            self.precompute_eig(L)

        # train iteration
        # 1 iteration same as 1 layer

        MIN_NUM = 10**-10
        X = torch.zeros_like(Y)

        for ite in range(self.layer):
            # lambda = 10^lambda_ : always > 0, learnable across orders of magnitude
            lambda_ite = 10.0 ** self.lambda_[ite]

            taps = self._taps(self.d_h[ite])

            # layer-specific Sobolev operator (L + eps_ite I)^beta_ite
            S = self._sobolev(ite)

            # temporal FIR kernel w (length L+1): x_{c+L} - sum_t taps[t] x_{c+L-t}
            w = torch.cat([-taps.flip(0), taps.new_ones(1)])

            if(ite==0):
                delta_X = -1*self._nabla(X,w,lambda_ite,Y,J,S)

            # Stepsize decision
            nablax = self._nabla(X,w,lambda_ite,Y,J,S)
            tau = -1 * self._matrix_product(delta_X,nablax)/(self._matrix_product(delta_X,self._nabla(delta_X,w,lambda_ite,Y,J,S)+Y)+MIN_NUM)
            # Search direction updating
            X_next = (X + tau*delta_X)
            nablaXnext = self._nabla(X_next,w,lambda_ite,Y,J,S)
            gamma = self._matrix_product(nablaXnext,nablaXnext)/(self._matrix_product(nablax,nablax)+MIN_NUM)
            delta_X = (-1*nablaXnext + gamma * delta_X)
            X = X_next

        out = X[:, :, self.n_taps-1+pad:]
        return out.squeeze(0) if squeeze else out


