
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.methods.proposed_batch_model import Batch_Model


class Online_Model(nn.Module):
    """Online time-varying graph-signal reconstruction using the unrolling batch model.

    Frame k is reconstructed by applying the batch model to the window
    [n_taps most recent reconstructions (treated as fully observed), current observation Y[:, k]],
    i.e. exactly the frames the temporal FIR touches. No future information is used.
    Before n_taps frames of history exist the window is zero-padded and those slots are marked
    unobserved, so early frames use whatever history is already there.

    The batch model's weights (lambda_, eps_, beta_, d_h) are trained for this recursion
    (train_from_datasets_online.py). The reconstructed history is detached, so gradients flow only
    through the current frame -- past reconstructions are already committed at deployment.
    """
    def __init__(self,layer:int,n_taps: int= 5,trained_model_path:str | None=None):
        """
        Parameters
        ----------
        layer : int
            depth of layers (unrolling iterations)
        n_taps : int
            number of learnable FIR taps (L in the paper)
        trained_model_path : str
            weights to load into the batch model. If None, parameters are random.
        """
        super().__init__()

        self.n_taps = n_taps
        self.layer = layer
        self.model = Batch_Model(layer=self.layer,n_taps=self.n_taps)

        if(trained_model_path is not None):
            self.model.load_state_dict(torch.load(trained_model_path, map_location='cpu'))

    def forward(self,Y,J,L):
        dev = self.model.lambda_[0].device
        dt = torch.get_default_dtype()
        Y = torch.as_tensor(Y, dtype=dt, device=dev)
        J = torch.as_tensor(J, dtype=dt, device=dev)
        L = torch.as_tensor(L, dtype=dt, device=dev)

        squeeze = (Y.dim() == 2)
        if squeeze:
            Y = Y.unsqueeze(0); J = J.unsqueeze(0)
        B, N, M = Y.shape
        if L.dim() == 2:
            L = L.unsqueeze(0).expand(B, -1, -1)

        X = torch.zeros(B, N, M, dtype=dt, device=dev)

        # L is constant across frames -> eigendecompose once and reuse for every frame's Sobolev op
        self.model.precompute_eig(L)

        for k in range(M):
            lo = max(k - self.n_taps, 0)
            Y_win = torch.cat([X[:, :, lo:k].detach(), Y[:, :, k:k+1]], dim=2)   # (B, N, <=T+1)
            J_win = torch.cat([torch.ones(B, N, k - lo, dtype=dt, device=dev), J[:, :, k:k+1]], dim=2)
            # checkpointing: the per-frame graphs of all M frames would otherwise hold the
            # intermediates of every unrolling layer and OOM. Passthrough under no_grad.
            recon = checkpoint(self.model.forward, Y_win, J_win, L, precomputed_eig=True, use_reentrant=False)
            X[:, :, k] = recon[:, :, -1]

        return X.squeeze(0) if squeeze else X
