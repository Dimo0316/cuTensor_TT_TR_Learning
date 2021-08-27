import jax.numpy as np
import  tensornetwork as tn
import time
from jax import random
key = random.PRNGKey(0)

#random 4 tensor
for n in range(5,55,5):
	A1 = random.normal(key, (n,n,n))
	A2 = random.normal(key, (n,n,n))
	A3 = random.normal(key, (n,n,n,n))
	A4 = A2

	tn.ncon([A1,A3,A2,A4],[(1,2,3),(4,-2,5,3),(1,5,-1),(2,4,-3)],backend='jax')

	t1 = time.time()
	tn.ncon([A1,A3,A2,A4],[(1,2,3),(4,-2,5,3),(1,5,-1),(2,4,-3)],backend='jax')
	t2 = time.time()
	print("\n")
	print("time for one site: ", str(t2-t1))


