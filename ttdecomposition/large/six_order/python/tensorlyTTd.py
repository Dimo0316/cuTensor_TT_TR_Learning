import numpy as np
import time
import tensorly as tl
import torch
from tensorly.decomposition import matrix_product_state
# from tensorly import tt_to_tensor
# from tensorly.contrib.decomposition import matrix_product_state_cross

calError = False
if torch.cuda.is_available():
    device = torch.device("cuda")
tl.set_backend('pytorch')

def genTTensor(k, i, datatype):
    np.random.seed(1)
    G=[0 for x in range(0,7)]
    G[6] = np.random.rand(k[5],i*k[6])
    G[5] = np.random.rand(k[4]*i,k[5])
    G[4] = np.random.rand(k[3]*i,k[4])
    G[3] = np.random.rand(k[2]*i,k[3])
    G[2] = np.random.rand(k[1]*i,k[2])
    G[1] = np.random.rand(k[0]*i,k[1])

    G[5] = np.matmul(G[5],G[6]).reshape(k[4],i*i*k[6])
    G[4] = np.matmul(G[4],G[5]).reshape(k[3],i*i*i*k[6])
    G[3] = np.matmul(G[3],G[4]).reshape(k[2],i*i*i*i*k[6])
    G[2] = np.matmul(G[2],G[3]).reshape(k[1],i*i*i*i*i*k[6])
    
    X = np.matmul(G[1],G[2]).reshape(i,i,i,i,i,i).astype(datatype)
    return X


def recover(G, k):
    G[4] = G[4].reshape(-1,k[5]).cpu()
    G[5] = G[5].reshape(k[5],-1).cpu()
    G[4] = np.matmul(G[4],G[5]).reshape(k[4],i*i*k[6])

    G[3] = G[3].reshape(-1,k[4]).cpu()
    G[3] = np.matmul(G[3],G[4]).reshape(k[3],i*i*i*k[6])

    G[2] = G[2].reshape(-1,k[3]).cpu()
    G[2] = np.matmul(G[2],G[3]).reshape(k[2],i*i*i*i*k[6])

    G[1] = G[1].reshape(-1,k[2]).cpu()
    G[1] = np.matmul(G[1],G[2]).reshape(k[1],i*i*i*i*i*k[6])

    G[0] = G[0].reshape(-1,k[1]).cpu()
    X = np.matmul(G[0],G[1]).reshape(i,i,i,i,i,i)
    return X


calTimes = 10
if(calError):
    calTimes = 1

for i in range(40, 44, 4):
    print("tensor size", i)

    # X = tl.tensor(np.random.random((i, i, i)))
    rank = 4
    if i > 24:
        rank = 8
    ranks = [1, rank, rank, rank, rank, rank, 1]
    X1 = genTTensor(ranks, i, np.float32)

    time_initial_start = time.time()
    X = tl.tensor(X1, device='cuda:0')
    time_initial_end = time.time()
    print("initial and transfer data time:" , time_initial_end - time_initial_start)

    factors = matrix_product_state(X, ranks)
    time_start = time.time()
    for j in range(0, calTimes):
        print("cal tiems", j)
        factors = matrix_product_state(X, ranks)
        for index in range(0, 6):
            factors[index].cpu()
    time_end = time.time()
    # print(factors)

    if(calError):
        TempX = recover(factors, ranks).cpu()
        X = X.cpu()
        # TempX = tt_to_tensor(factors)
        err = np.linalg.norm(X-TempX)/np.linalg.norm(X)
        print("err:",err)
    

    f = open('cp.txt', 'a')
    f.write(str(i))
    f.write('   ')
    f.write(str((time_end - time_start)/calTimes))
    f.write('\n')
    print('time cost ', (time_end - time_start)/calTimes +  time_initial_end - time_initial_start, 's')
