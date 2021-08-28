import numpy as np
import time
import tntorch as tn
import tensornetwork as tnk
import torch
import numpy as np
import torch
import tntorch as tn
torch.set_default_dtype(torch.float64)
import time


def _full_rank_tt(data):  # Naive TT formatting, don't even attempt to compress
    data = data.to(torch.get_default_dtype())
    shape = data.shape
    result = []
    N = data.dim()
    data = torch.Tensor(data) if type(data) is not torch.Tensor else data
    device = data.device
    resh = torch.reshape(data, [shape[0], -1])
    for n in range(1, N):
        if resh.shape[0] < resh.shape[1]:
            result.append(torch.reshape(torch.eye(resh.shape[0]).to(device), [resh.shape[0] // shape[n - 1],
                                                                       shape[n - 1], resh.shape[0]]))
            resh = torch.reshape(resh, (resh.shape[0] * shape[n], resh.shape[1] // shape[n]))
        else:
            result.append(torch.reshape(resh, [resh.shape[0] // shape[n - 1],
                                                   shape[n - 1], resh.shape[1]]))
            resh = torch.reshape(torch.eye(resh.shape[1]).to(device), (resh.shape[1] * shape[n], resh.shape[1] // shape[n]))
    result.append(torch.reshape(resh, [resh.shape[0] // shape[N - 1], shape[N - 1], 1]))
    return result


def _broadcast(a, b):
    if a.shape == b.shape:
        return a, b
    elif a.dim() != b.dim():
        raise ValueError('Cannot broadcast: lhs has {} dimensions, rhs has {}'.format(a.dim(), b.dim()))
    result1 = a.repeat(*[int(round(max(sh2/sh1, 1))) for sh1, sh2 in zip(a.shape, b.shape)])
    result2 = b.repeat(*[int(round(max(sh1 / sh2, 1))) for sh1, sh2 in zip(a.shape, b.shape)])
    return result1, result2


def _core_kron(a, b):
    # return torch.reshape(torch.einsum('iaj,kal->ikajl', (a, b)), [a.shape[0]*b.shape[0], -1, a.shape[2]*b.shape[2]])  # Seems slower
    c = a[:, None, :, :, None] * b[None, :, :, None, :]
    c = c.reshape([a.shape[0] * b.shape[0], -1, a.shape[-1] * b.shape[-1]])
    return c
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

calTimes = 1
writeFile = False
calError = True

for i in range(8, 32, 4):
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
