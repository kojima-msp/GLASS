import random

import numpy as np
import torch
from torch import nn, optim


class Inpainting_DAU(nn.Module):
    """TIME-VARYING GRAPH SIGNAL INPAINTING VIA UNROLLING NETWORKS
    """
    def __init__(self,layer:int,default_lambda: float | None=1.0,P :int = 2,Q :int = 5):
        """__init__
        Parameters
        ----------
        layer : int
            depth of layers
        default_lambda : float
            init lambda (default:1.0)
        P : int
            Edge tap length (default:2)
        Q : int
            Time tap length (default:5)
        """
        super().__init__()
        if default_lambda is None:
            default_lambda = random.random()
        self.P = P
        self.Q = Q
        
        # trainable params
        # lambda_:1 x len(layer)
        self.lambda_ = nn.ParameterList([nn.Parameter(torch.tensor([default_lambda])) for _ in range(layer)])

        # g_q:len(layer) x T
        self.g_q = nn.ParameterList([nn.Parameter(torch.tensor([random.random() for _ in range(Q-1)])) for __ in range(layer)])
        # h_p:len(layer) x T
        self.h_p = nn.ParameterList([nn.Parameter(torch.tensor([random.random() for _ in range(P-1)])) for __ in range(layer)])

        self.layer = layer

        self.relu = torch.nn.ReLU()

    def _nabla(self,matrix,lambda_,hL_G,gL_T):
        return torch.mul(self.J,matrix) - self.Y + lambda_* hL_G @ matrix @ gL_T

    def _matrix_product(self,matrix1,matrix2):
        return torch.sum(torch.mul(matrix1,matrix2))

    def forward(self,Y,J,DG):
        """
        Parameters
        ----------
        Y : Array(NxM)
            sampled data
        J : Array(NxM)
            sampling operator
            each element must have 0 or 1
        DG : Array(MxN)
            

        Returns
        ----------
        X : Array(NxM)
            reconstructed signal
        """
        MIN_NUM = 10**-10

        DT = np.eye(Y.shape[1],k=-1) - np.eye(Y.shape[1])
        DT[0,-1] = 1

        self.N = Y.shape[0]
        self.M = Y.shape[1]
        # follow the default dtype (np.eye above is float64, while DataLoader arrays are float32)
        _dt = torch.get_default_dtype()
        self.J = torch.tensor(J, dtype=_dt)
        self.LG = torch.tensor(DG.T @ DG, dtype=_dt)
        self.LT = torch.tensor(DT.T @ DT, dtype=_dt)
        self.DT = torch.tensor(DT, dtype=_dt)
        self.DG = torch.tensor(DG, dtype=_dt)

        self.Y = torch.tensor(Y, dtype=_dt)

        # h(L_G) = L_G + sum_p h_p L_G^p and g(L_T) = L_T + sum_q g_q L_T^q are polynomials of
        # the operators, so the powers are matrix powers and are the same in every layer
        LG_pow = [torch.linalg.matrix_power(self.LG, p) for p in range(2, self.P+1)]
        LT_pow = [torch.linalg.matrix_power(self.LT, q) for q in range(2, self.Q+1)]

        # train iteration
        # 1 iteration same as 1 layer

        for ite in range(self.layer):
            # initialization
            self.lambda_[ite].data= self.relu(self.lambda_[ite].data)
            self.h_p[ite].data= self.relu(self.h_p[ite].data)
            self.g_q[ite].data= self.relu(self.g_q[ite].data)

            # init hL_G
            hL_G = self.LG.clone()
            for p in range(2,self.P+1):
                hL_G = hL_G + self.h_p[ite][p-2] * LG_pow[p-2]
            # init gL_T
            gL_T = self.LT.clone()
            for q in range(2,self.Q+1):
                gL_T = gL_T + self.g_q[ite][q-2] * LT_pow[q-2]

            # when first iteration
            if(ite==0):
                X = torch.zeros(self.N,self.M)
                delta_X = -1*self._nabla(X,self.lambda_[ite],hL_G,gL_T)
            # --------------------

            # Stepsize decision
            nablax = self._nabla(X,self.lambda_[ite],hL_G,gL_T)
            tau = -1 * self._matrix_product(delta_X,nablax)/(self._matrix_product(delta_X,self._nabla(delta_X,self.lambda_[ite],hL_G,gL_T)+self.Y)+MIN_NUM)
            # --------------------

            # Search direction updating
            X_next = (X + tau*delta_X)
            nablaXnext = self._nabla(X_next,self.lambda_[ite],hL_G,gL_T)
            gamma = self._matrix_product(nablaXnext,nablaXnext)/(self._matrix_product(nablax,nablax)+MIN_NUM)
            delta_X = (-1*nablaXnext + gamma * delta_X)
            X = X_next
        
        return X

    def forward_unsupervised(self,Y,J,DG):
        self.train()
        lr = 0.01
        Epochs = 100
        step_size= 10
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        criterion = torch.nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.9)
        optimizer.param_groups[0]['capturable'] = True

        loss_min = float('inf')
        X = torch.zeros(Y.shape[0],Y.shape[1])
        for _ in range(Epochs):
            out = self.forward(Y,J,DG)
            loss = criterion(torch.mul(out,torch.tensor(J)),torch.tensor(Y).to(device))
            optimizer.zero_grad()
            
            if(loss.cpu().detach().numpy()<loss_min):
                loss_min = loss.cpu().detach().numpy()
                X = out
            
            loss.backward()
            optimizer.step()
            scheduler.step()

        return X

if(__name__=='__main__'):
    pass
    