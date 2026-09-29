import numpy as np
import torch
from torch import nn
from torch_geometric.nn import ChebConv, GCNConv


def _edges_from_laplacian(L):
    """Recover the (weighted) adjacency edges from a graph Laplacian.
    The off-diagonal of -L holds the edge weights, which PyG re-normalizes."""
    A = -L.clone()
    A.fill_diagonal_(0.0)
    idx = torch.nonzero(A, as_tuple=False)        # (E, 2)
    edge_index = idx.t().contiguous().long()      # (2, E)
    edge_weight = A[idx[:, 0], idx[:, 1]]         # (E,)
    return edge_index, edge_weight


class _PerFrameGNN(nn.Module):
    """Apply a standard GNN to every time frame independently (no temporal modeling).

    Each frame is restored from its observed values and the sampling mask, i.e. the
    temporal correlation that the proposed method exploits is deliberately ignored.
    Node features per frame are ``[observed value, mask]`` and the network regresses
    the restored value at each node. The M frames are processed at once as a batch of
    M disjoint copies of the same graph.
    """

    def forward(self, Y, J, L):
        """Y, J : (N, M) observed signal / 0-1 mask,  L : (N, N) graph Laplacian.
        Returns the restored signal (N, M)."""
        Y = torch.as_tensor(np.asarray(Y), dtype=torch.get_default_dtype())
        J = torch.as_tensor(np.asarray(J), dtype=torch.get_default_dtype())
        L = torch.as_tensor(np.asarray(L), dtype=torch.get_default_dtype())
        N, M = Y.shape

        # standardize by the observed-value statistics; de-standardized at the end
        obs = Y[J > 0]
        mu, sd = obs.mean(), obs.std() + 1e-8
        Yn = ((Y - mu) / sd) * J

        edge_index, edge_weight = _edges_from_laplacian(L)
        # replicate the graph for every frame (block-diagonal batch); node m*N+n is
        # node n of frame m, so frame m's edges are shifted by m*N.
        offsets = (torch.arange(M, device=edge_index.device) * N).view(1, M, 1)
        batch_edge_index = (edge_index.unsqueeze(1) + offsets).reshape(2, -1)
        batch_edge_weight = edge_weight.repeat(M)

        # frame-major node features: node (m*N + n) -> [Yn[n, m], J[n, m]]
        x = torch.stack([Yn, J], dim=-1).permute(1, 0, 2).reshape(M * N, 2)

        for conv in self.convs[:-1]:
            x = self.relu(conv(x, batch_edge_index, batch_edge_weight))
        x = self.convs[-1](x, batch_edge_index, batch_edge_weight)  # (M*N, 1)
        gnn = x.view(M, N).t()                       # (N, M), in the standardized space
        # the GNN predicts a correction on top of the masked input (residual)
        return (Yn + gnn) * sd + mu


class GCN(_PerFrameGNN):
    """Per-frame GCN baseline (Kipf & Welling, ICLR 2017)."""

    def __init__(self, in_channels=2, hidden=32, n_layers=2):
        super().__init__()
        dims = [in_channels] + [hidden] * (n_layers - 1) + [1]
        self.convs = nn.ModuleList([GCNConv(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.relu = nn.ReLU()


class ChebNet(_PerFrameGNN):
    """Per-frame ChebNet baseline (Defferrard et al., NeurIPS 2016)."""

    def __init__(self, in_channels=2, hidden=32, n_layers=2, cheb_K=3):
        super().__init__()
        dims = [in_channels] + [hidden] * (n_layers - 1) + [1]
        self.convs = nn.ModuleList([ChebConv(dims[i], dims[i + 1], K=cheb_K) for i in range(len(dims) - 1)])
        self.relu = nn.ReLU()