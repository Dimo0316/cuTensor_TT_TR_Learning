import numpy as np
import time
import tensorly as tl
import torch
from tensorly.decomposition import matrix_product_state
from tensorly.contrib.decomposition import matrix_product_state_cross

if torch.cuda.is_available():
    device = torch.device("cuda")
tl.set_backend('pytorch')
calTimes = 10
for i in range(900, 1100, 100):
    print(i)

    # k = 100
    # a = np.random.rand(k, k)
    # a = a.astype(np.float32)
    # b = np.random.rand(k, k)
    # b = b.astype(np.float32)
    # c = np.random.rand(k, k * k)
    # c = c.astype(np.float32)
    # h_X = np.dot(c, (np.kron(a, b).transpose()))
    # X = tl.tensor(h_X.reshape(k, k, k), dtype=tl.float32)

    X = tl.tensor(np.random.random((i, i, i)))
    # X = tl.tensor(np.random.random((i, i, i)), device='cuda:0')
    rank = [1, i//10//8*8, i//10//8*8, 1]
    print(rank)
    time_start = time.time()
    for j in range(0, calTimes):
        factors = matrix_product_state_cross(X, rank)
    time_end = time.time()
    f = open('cp.txt', 'a')
    f.write(str(i))
    f.write('   ')
    f.write(str((time_end - time_start)/calTimes))
    f.write('\n')
    print('time cost ', (time_end - time_start)/calTimes, 's')
