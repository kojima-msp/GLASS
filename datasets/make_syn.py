import itertools
import os
import random

import numpy as np
import scipy
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors

# setting
SIGNAL_LEVEL = 1        # signal smoothness level
RANGE_MAX = 100         # node coordinate range [0,RANGE_MAX)
K = 5                   # KNN parameter
N_nodes = 100           # number of nodes
N_times = 600           # number of time series
N_GRAPHS = 15           # data_00.npz ... data_{N_GRAPHS-1}.npz
MASTER_SEED = 20240101  # seed

def gen_graph():
    """
    Returns
    ----------
    L : Array(NxN)
        Graph Laplacian
    node_set : Array(N_nodesx2)
        set of nodes
    W : Array(NxN)
        weight matrix
    E : Array(|E|x2)
        Edge list
    DeltaG : Array(|E|xN_nodes)
        transpose of the incidence matrix
    """

    # select N_nodes samples uniformly from the integer grid
    node_set = np.array(random.sample(list(itertools.product(list(range(RANGE_MAX)), list(range(RANGE_MAX)))),N_nodes))

    # search kNN graph (k=5)
    nbrs = NearestNeighbors(n_neighbors = K+1,algorithm='ball_tree').fit(node_set)
    A = nbrs.kneighbors_graph(node_set).toarray()
    # delete self-loops
    A = A - np.diag(np.ones(A.shape[0]))
    # convert to an undirected graph
    A = np.maximum(A, A.T)

    E = np.array(np.where(np.triu(A) == 1)).T

    # make DeltaG matrix (for DAU method)
    DeltaG = np.zeros((E.shape[0],N_nodes))
    for i,axis in enumerate(E):
        DeltaG[i,axis[1]] = 1
        DeltaG[i,axis[0]] = -1

    # calc graph operator
    # weights
    W = np.zeros_like(A)
    for node_pair in E:
        W[node_pair[0],node_pair[1]] = 1/distance.euclidean(node_set[node_pair[0]],node_set[node_pair[1]])**2
        W[node_pair[1],node_pair[0]] = W[node_pair[0],node_pair[1]]
    W = W/W.max()
    # degree
    D = np.diag(np.ravel(W.sum(1)), 0)
    L = D - W

    return L,node_set, W, E, DeltaG

def gen_signal(L,N_vertex,N_times):

    # eigenvalue decomposition
    lambda_,U = scipy.linalg.eigh(L)
    lambda_ = np.where(lambda_<0,0,lambda_) # del caluclation error

    # L_m : L^{-1/2}, with the null space left at zero
    nonzero = lambda_ > 1e-12
    lambdaHalfInv = np.zeros_like(lambda_)
    lambdaHalfInv[nonzero] = 1/np.sqrt(lambda_[nonzero])
    L_m = U @ np.diag(lambdaHalfInv) @ U.T

    # gen signals
    tv_signal = np.zeros((N_vertex,N_times))

    ftmp = U.T @ np.random.randn(N_vertex)
    cutoff = int(0.1 * N_vertex)   # weaken all but the lowest 10% of the graph frequencies
    ftmp[cutoff:] = ftmp[cutoff:]/100
    ftmp = U @ ftmp
    tv_signal[:,0] = ftmp/np.linalg.norm(ftmp)*100

    for n_time in range(1,N_times):
        f = np.random.randn(N_vertex)
        tv_signal[:,n_time] = tv_signal[:,n_time-1] + L_m @ (SIGNAL_LEVEL * f / np.linalg.norm(f))

    return tv_signal

if(__name__=='__main__'):
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Synthetic')
    for i in range(N_GRAPHS):
        # seed graph & signal RNGs deterministically per graph
        seed = int(np.random.SeedSequence([MASTER_SEED, i, 0]).generate_state(1)[0])
        random.seed(seed)
        np.random.seed(seed)

        # gen graph
        L, node_set, W, E, DeltaG = gen_graph()

        # gen signal
        tv_signal = gen_signal(L,N_nodes,N_times)

        # the sampling ratio and the noise level are applied at load time (src/utils/data.py)
        rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, i, 1]))
        uniform = rng.random(tv_signal.shape)
        noise = tv_signal.std() * rng.standard_normal(tv_signal.shape)   # scaled by the signal std

        np.savez(os.path.join(out_dir, f'data_{str(i).zfill(2)}'),
                 groundtruth=tv_signal, L=L, node=node_set, W=W, E=E, DeltaG=DeltaG,
                 uniform=uniform, noise=noise)