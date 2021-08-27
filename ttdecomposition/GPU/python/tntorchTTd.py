import numpy as np
import time
import tntorch as tn
import tensornetwork as tnk
import torch

tnk.set_default_backend("pytorch")

def metrics(t2, full):
    print(t2)
    print('Compression ratio: {}/{} = {:g}'.format(full.numel(), t2.numel(), full.numel() / t2.numel()))
    print('Relative error:', tn.relative_error(full, 2))
    print('RMSE:', tn.rmse(full, t2))
    print('R^2:', tn.r_squared(full, t2))

if torch.cuda.is_available():
    device = torch.device("cuda")

calTimes = 1
writeFile = False
calError = True

for i in range(800, 1000, 100):
    print(i)

    r = 56
    if(i>600):
        r = i//10//8*8

    k = [1, r, r, 1]
    G=[0 for x in range(0,4)]
    G[3] = torch.rand(r,i,1)
    G[2] = torch.rand(r,i,r)
    G[1] = torch.rand(1,i,r)
    G[2] = tnk.ncon([G[2],G[3]],[(-1,-2,1),(1,-3,-4)]).reshape(r, i*i, 1)
    X = tnk.ncon([G[1],G[2]],[(-1,-2,1),(1,-3,-4)]).reshape(i,i,i)
    X = torch.reshape(torch.squeeze(X),(i,i,i)).cuda()

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
