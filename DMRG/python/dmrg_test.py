import tensornetwork as tn
import numpy as np
import tensorflow as tf
import time
tf.enable_v2_behavior()
GPUController = True
if GPUController:
  backend = 'jax'
else:
  backend = 'numpy'

D = 4
d = 2
N = 200
print(N)
mps = tn.FiniteMPS.random(d = [d]*N, D=[D]*(N-1), dtype=np.float64, backend=backend)
Jz = np.ones(N-1)
Jxy = np.ones(N-1)
Bz = np.zeros(N)

t1 = time.time()
mpo = tn.FiniteXXZ(Jz = Jz, Jxy = Jxy, Bz = Bz, dtype=np.float64, backend=backend)
dmrg = tn.FiniteDMRG(mps,mpo)
E=dmrg.run_one_site(num_sweeps=5, verbose=1)
t2 = time.time()

run_time = t2-t1
print ('cost time is：',run_time)