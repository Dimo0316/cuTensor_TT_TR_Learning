import numpy as np
import time
import tntorch as tn
import tensornetwork as tnk
import torch

tnk.set_default_backend("pytorch")

def genTTensor(k, i):
    np.random.seed(1)
    G=[0 for x in range(0,7)]
    G[6] = torch.rand(k[5],i*k[6])
    G[5] = torch.rand(k[4]*i,k[5]) 
    G[4] = torch.rand(k[3]*i,k[4]) 
    G[3] = torch.rand(k[2]*i,k[3]) 
    G[2] = torch.rand(k[1]*i,k[2]) 
    G[1] = torch.rand(k[0]*i,k[1]) 

    G[5] = tnk.ncon([G[5],G[6]],[(-1,1),(1,-2)]).reshape(k[4], -1)
    G[4] = tnk.ncon([G[4],G[5]],[(-1,1),(1,-2)]).reshape(k[3], -1)
    G[3] = tnk.ncon([G[3],G[4]],[(-1,1),(1,-2)]).reshape(k[2], -1)
    G[2] = tnk.ncon([G[2],G[3]],[(-1,1),(1,-2)]).reshape(k[1], -1)
    G[1] = tnk.ncon([G[1],G[2]],[(-1,1),(1,-2)]).reshape(i,i,i,i,i,i)

    return G[1]

def metrics(t2, full):
    print(t2)
    print('Compression ratio: {}/{} = {:g}'.format(full.numel(), t2.numel(), full.numel() / t2.numel()))
    print('Relative error:', tn.relative_error(full, 2))
    print('RMSE:', tn.rmse(full, t2))
    print('R^2:', tn.r_squared(full, t2))

if torch.cuda.is_available():
    device = torch.device("cuda")

calTimes = 5
writeFile = False
calError = True

for i in range(8, 44, 4):
    print(i)
    r = 4
    if(i>24):
        r = 8
    k = [1, r, r, r, r, r, 1]
 
    X = genTTensor(k, i)
    X = torch.reshape(torch.squeeze(X),(i,i,i,i,i,i)).cuda()

    # X = torch.tensor(np.random.random((i, i, i)))
    # X = torch.tensor(np.random.random((i, i, i)), device='cuda:0')
    # rank = [1, i//10//8*8, i//10//8*8, 1]
    # rank_single = i//10//8*8

    time_start = time.time()
    for j in range(0, calTimes):
        t = tn.Tensor(X, ranks_tt=r).torch()
    time_end = time.time()
    
    if(writeFile):
        f = open('tntorch.txt', 'a')
        f.write(str(i))
        f.write('   ')
        f.write(str((time_end - time_start)/calTimes))
        if(calError):
            err = torch.norm(X-t)/torch.norm(X)
            print(err)
            f.write('   ')
            f.write(str(err))
        f.write('\n')

    if(calError):
        err = torch.norm(X-t)/torch.norm(X)
        print(err)

    # metrics(t, X)
    print('time cost ', (time_end - time_start)/calTimes, 's')


#
# X = torch.Tensor(h_X.reshape(k, k, k))
# print("start TT decomposition")
# a = time.time()
# for i in range(0, 5):
#     t = tnt.Tensor(X)
#     t.round_tt(eps=1e-6)
#     #factors = matrix_product_state_cross(X, [1, 120, 120, 1])
# b = time.time()
# print("time:", (b-a)/5.0)
# metrics(t, X)
