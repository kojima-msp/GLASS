import numpy as np
import torch
from torch import nn, optim

from src.methods.proposed_batch_model import Batch_Model


class Unsupervised_Model(nn.Module):
    """Unsupervised Time-varying Graph Signal Batch Reconstruction Model using unrolling

    Every parameter is trained on the target observation alone, so the method needs neither
    training data nor a ground truth.
    """
    def __init__(self,layer:int,n_taps: int= 5):
        """__init__
        Parameters
        ----------
        layer : int
            depth of layers
        n_taps : int
            number of learnable FIR taps (L in the paper, default:5)
        """
        super().__init__()

        self.n_taps = n_taps
        self.layer = layer
        self.model = Batch_Model(layer=self.layer,n_taps=self.n_taps)

    def forward(self,Y,J,L,Epochs=100,lr=1e-3):

        criterion = torch.nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

        loss_min = np.inf

        for _ in range(Epochs):
            out = self.model.forward(Y,J,L)
            loss = criterion(torch.mul(out,torch.tensor(J)),torch.tensor(Y))
            if(loss_min>loss.cpu().detach().numpy()):
                loss_min = loss.cpu().detach().numpy()
                X = out.cpu().detach().numpy()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        return X
