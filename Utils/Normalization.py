import numpy as np
import torch

def normalize_adj(adj):
    length = adj.shape[0]
    tilde_adj = adj + np.identity(length) # adjaceny matrix (A) tilde
    deg = np.sum(tilde_adj, axis=1)
    norm_deg = np.power(deg ,-0.5)
    tilde_deg = np.diag(norm_deg) # degree matrix (D) tilde
    nor_adj = np.matmul(np.matmul(tilde_deg, tilde_adj) ,tilde_deg)

    return nor_adj

def normalization(x :torch.Tensor, axis :int = 0, ntype :str = None) -> torch.Tensor:
    if ntype == None:
        print("ntype is missed -- original tensor is returned")
        return x
    elif ntype == 'standardization':
        return ( x -x.mean(axis=axis) ) /x.std(axis=axis)
    elif ntype == 'min-max':
        return (x - x.min() ) /(x.max() - x.min())