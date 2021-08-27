import numpy as np
import time
import tensorly as tl
import torch
from tensorly.decomposition import matrix_product_state
from tensorly.contrib.decomposition import tensor_train_cross

if torch.cuda.is_available():
    device = torch.device("cuda")
tl.set_backend('pytorch')
calTimes = 10
for i in range(600, 700, 100):
    print(i)
    # X = tl.tensor(np.random.random((i, i, i)))
    X = tl.tensor(np.random.random((i, i, i)), device='cuda:0')
    time_start = time.time()
    rank = [1, i//10//8*8, i//10//8*8, 1]
    for j in range(0, calTimes):
        factors = tensor_train_cross(X, rank)
    time_end = time.time()
    f = open('cp.txt', 'a')
    f.write(str(i))
    f.write('   ')
    f.write(str((time_end - time_start)/calTimes))
    f.write('\n')
    print('time cost ', (time_end - time_start)/calTimes, 's')
