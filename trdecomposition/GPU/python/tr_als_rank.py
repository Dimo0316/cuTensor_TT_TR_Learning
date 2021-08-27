# -*- coding: UTF-8 -*-

import torch
import tensorly as tl
from time import *
tl.set_backend('pytorch')



def computZ(factors):
    Z = factors[0]#0
    leng = len(factors)-2
    for i in range(leng):#1 - d-2
       Z = torch.tensordot(Z,factors[i+1],dims = ([-1],[0]))
    Z = torch.tensordot(Z,factors[-1],dims = ([-1,0],[0,2]))
    return  Z


def tensoring(input_tensor, rank, Maxiter = 50,Maxiter2 =2000, delt = 10e-6, verbose=True, init = "Norm"):
    """Tensor Ring decomposition via ALS
        Decomposes `input_tensor` into a sequence of order-3 tensors (factors)
    Parameters
    ----------
    input_tensor : tensorly.tensor
    rank : {int, int list}
            maximum allowable MPS rank of the factors
            if int, then this is the same for all the factors
            if int list, then rank[k] is the rank of the kth factor
    
     : boolean, optional
            level of verbosity
    Returns
    -------
    factors : Tensor factors
              order-3 tensors of the TR decomposition
    References
    ----------
    .. [1] Qibin Zhao, Guoxu Zhou, Shengli Xie, Liqing Zhang, Andrzej Cichocki. "Tensor Ring Decomposition".
    """
 
    
    # Check user input for errors
    tensor_size = input_tensor.shape
    n_dim = len(tensor_size)

    if isinstance(rank, int):
        rank = [rank] + [rank] * (n_dim-1) + [rank]
    elif n_dim + 1 != len(rank):
        rank = rank + rank[0]
        message = 'Provided incorrect number of ranks. Should verify len(rank) == tl.ndim(tensor)+1, but len(rank) = {} while tl.ndim(tensor) + 1  = {}'.format(
            len(rank), n_dim + 1)
        raise(ValueError(message))

    # Make sure it's not a tuple but a list
    rank = list(rank)
    T = input_tensor
    factors = [None] * n_dim
    optimizer = []
    
    if init == "Norm":
        # Initialization
        for k in range(n_dim):
            size = [rank[k],tensor_size[k],rank[k+1]]
            factor = torch.randn(size,requires_grad = True)
            factors[k] = factor

    
    elif init == "SVD":
        # Initialization2
        unfolding = input_tensor
        n_row = int(tensor_size[0])
        unfolding = tl.reshape(unfolding, (n_row, -1))
        current_rank = rank[0] * rank[1]
        U, S, V = tl.partial_svd(unfolding, current_rank)
        factors[0] = tl.reshape(U, (tensor_size[0], rank[0],  rank[1])).permute(1,2,0)
        unfolding = torch.tensordot(S, V, dims=([1], [0]))
        unfolding = tl.reshape(unfolding, (rank[0] , rank[1], -1)).permute(1,2,0)
        
        
        for h in range(n_dim - 2):
            k =h + 1
            # Reshape the unfolding matrix of the remaining factors
            n_row = int(rank[k] * tensor_size[k])
            unfolding = tl.reshape(unfolding, (n_row, -1))
    
            # SVD of unfolding matrix
            (n_row, n_column) = unfolding.shape
            current_rank = min(n_row, n_column, rank[k + 1])
            U, S, V = tl.partial_svd(unfolding, current_rank)
            rank[k + 1] = current_rank
    
            # Get kth MPS factor
            factors[k] = tl.reshape(U, (rank[k], tensor_size[k], rank[k + 1]))
            unfolding = torch.tensordot(S ,V, dims=([1],[0]))
            
        factors[n_dim-1] = tl.reshape(unfolding, (rank[n_dim-1], tensor_size[k], rank[n_dim]))
    else:
        exit(0)
        
    for k in range(n_dim):
        optimizer = optimizer + [torch.optim.Adam({factors[k]}, lr=1e-1)]
    #TR_ALS
    loss_ep = 100
    iter = 0
    while (loss_ep > delt) and (iter < Maxiter):
        
        iter = iter + 1
        
        for k in range(n_dim):
            iter2 = 0
            loss_before = 0
            while (iter2 < Maxiter2):
                iter2 = iter2 + 1
                Z = computZ(factors)
                Loss = (torch.pow((T - Z),2)).sum()/(torch.pow((T),2)).sum()
                if torch.abs(Loss - loss_before)<10e-8:
                    break
                loss_before = Loss
                optimizer[k].zero_grad()
                Loss.backward()
                optimizer[k].step()
    
            loss_ep = (torch.pow((T - Z),2)).sum()/(torch.pow((T),2)).sum()
        
        if(verbose is True):
            print("loss_ep ")
            print(loss_ep)
            print(" iter ")
            print(iter)
 
    return factors


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    tensor = torch.randn([100,100,100], requires_grad = False)
    factors = tensoring(tensor, 56)