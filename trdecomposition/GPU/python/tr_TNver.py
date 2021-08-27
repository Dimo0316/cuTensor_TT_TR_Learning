import tensornetwork as tn
import numpy as np
import tensorly as tl
import tensorflow as tf
from tensorly import *
import torch
from time import *

tn.set_default_backend("pytorch")
calRecover = True

def repmat(a, m, n):
    a = np.asanyarray(a)
    ndim = a.ndim
    if ndim == 0:
        origrows, origcols = (1, 1)
    elif ndim == 1:
        origrows, origcols = (1, a.shape[0])
    else:
        origrows, origcols = a.shape
    rows = origrows * m
    cols = origcols * n
    c = a.reshape(1, a.size).repeat(m, 0).reshape(rows, origcols).repeat(n, 0)
    return c.reshape(rows, cols)


for i in range (100,500,100):
	print("===============",i,"================")
	n = i
	a=n
	b=n
	c=n
	m = 50
	k=[2,28,100,2]
# 	print(k)
# #generate tensor U
	
# 	G[3] = torch.rand(k[2],c*k[3]).cuda()
# 	G[2] = torch.rand(k[1]*b,k[2]).cuda()
# 	G[1] = torch.rand(a,k[0]*k[1]).cuda()

# 	G[2] = tn.ncon([G[2],G[3]],[(-1,1),(1,-2)]).reshape(k[1],b*c,k[3]).permute(0,2,1).reshape(k[1]*k[3],b*c)
	
# 	U = tn.ncon([G[1],G[2]],[(-1,1),(1,-2)]).reshape(a,b,c) 
	G=[0 for x in range(0,4)]
	G[1] = np.random.rand(i, m)
	G[2] = np.random.rand(i, m)
	G[3] = np.random.rand(m, i)
	one = np.ones((i,1))
	G[1] = np.kron(G[1],one)

	G[2] = repmat(G[2],i,1)
	G[2] = G[1]*G[2]
	X = np.matmul(G[2],G[3]).reshape(i,i,i)
	U = torch.from_numpy(X).to(torch.float32).cuda()
	# print(U)
	calTimes = 1
	
# tr decomposition
	begin_time = time()
	for tim in range(0,calTimes):
		print("-----------------",tim,"----------------")
		# G = [0 for x in range(0,4)]
		G[0] = torch.reshape(torch.squeeze(U),(a,b*c)).cuda()
		G[1], S, V = torch.svd(G[0])
		G[1] = torch.reshape(G[1][:,:(k[1]*k[0])],(a,k[0],k[1])).permute(1,0,2) #2*i*28
	
		G[0] = tn.ncon([(torch.diag(S))[:(k[1]*k[0]),:(k[1]*k[0])],(torch.t(V))[:(k[1]*k[0]),]],[(-1,1),(1,-2)]).reshape(k[0],k[1],b*c).permute(1,2,0).reshape(k[1]*b,c*k[3])
		G[2],S,V = torch.svd(G[0])
		# print("G2 size", G[2].size())
		G[2] = torch.reshape(G[2][:,:k[2]],(k[1],b,k[2]))

		G[3] = tn.ncon([(torch.diag(S))[:k[2],:k[2]],(torch.t(V))[:k[2],]],[(-1,1),(1,-2)]).reshape(k[2], c, k[3])
	
	# recover tesnro and calculate MSE
		if(calRecover):
			G[2] = tn.ncon([G[2].reshape(k[1]*b,k[2]),G[3].reshape(k[2], c*k[3])],[(-1,1),(1,-2)]).reshape(k[1], b*c, k[0]).permute(2,0,1)
			print(G[2].size())
			G[1] = tn.ncon([G[1].permute(1,0,2).reshape(a,k[1]*k[0]),G[2].reshape(k[0]*k[1],b*c)],[(-1,1),(1,-2)]).reshape(a, b, c)
			print(G[1].size())
			err=torch.norm(U - G[1])/torch.norm(U)
			print("err:", err.data)
			f = open('err.txt', 'a')
			f.write(str(i))
			f.write('   ')
			f.write(str(err))
			f.write('\n')
			f.close()

	end_time = time()
	run_time = (end_time-begin_time)/calTimes
	print ('cost time is：',run_time)

	print("=============== finish ================")
