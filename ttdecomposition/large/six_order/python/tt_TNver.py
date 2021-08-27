import tensornetwork as tn
import numpy as np
import tensorly as tl
from tensorly import *
import tensorflow as tf
import torch
from time import *
from tensornetwork.backends.pytorch import decompositions

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


for i in range (100,600,100):
	print("===============",i,"================")
	n = i
	a=n
	b=n
	c=n
	m=50
	r = 56
	if(n>600):
		r = n//10//8*8

	print(r)
	k=[1,r,r,1]
	print(k)
#generate tensor U[0]
	# G=[0 for x in range(0,4)]
	# G[3] = torch.rand(k[2],c,k[3]).cuda()
	# G[2] = torch.rand(k[1],b,k[2]).cuda()
	# G[1] = torch.rand(k[0],a,k[1]).cuda()
	
	# # G[2] = tn.ncon([G[2],G[3]],[(-1,1),(1,-2)])
	# # U = tn.ncon([G[1],G[2]],[(-1,1),(1,-2)]).reshape(a,b,c)
	# G[2] = tn.ncon([G[2],G[3]],[(-1,-2,1),(1,-3,-4)]).reshape(k[1], b*c, k[3])
	# U = tn.ncon([G[1],G[2]],[(-1,-2,1),(1,-3,-4)]).reshape(a,b,c)
	# U = torch.reshape(torch.squeeze(U),(a,b,c))
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

	calTimes = 1
	
# tt decomposition
	begin_time = time()
	for tim in range(0,calTimes):
		print("-----------------",tim,"----------------")
		# G = [0 for x in range(0,4)]
		G[0] = torch.reshape(torch.squeeze(U),(a,b*c)).cuda()
		# G[1], S, V = decompositions.svd(torch, G[0], 1)
		G[1], S, V = torch.svd(G[0])
		G[1] = torch.reshape(G[1][:,:k[1]],(k[0],a,k[1]))
		G[0] = tn.ncon([(torch.diag(S))[:k[1],:k[1]],(torch.t(V))[:k[1],]],[(-1,1),(1,-2)]).reshape(k[1]*b, c*k[3])
		# G[0] = torch.reshape(G[0],(k[1]*b, c*k[3]))
		G[2],S,V = torch.svd(G[0])
		G[2] = torch.reshape(G[2][:,:k[2]],(k[1],b,k[2]))
		G[3] = tn.ncon([(torch.diag(S))[:k[2],:k[2]],(torch.t(V))[:k[2],]],[(-1,1),(1,-2)]).reshape(k[2], c, k[3])
	
	# recover tesnro and calculate MSE
		if(calRecover):
			G[2] = tn.ncon([G[2].reshape(k[1]*b,k[2]),G[3].reshape(k[2], c*k[3])],[(-1,1),(1,-2)]).reshape(k[1], b*c*k[3])
			G[1] = tn.ncon([G[1].reshape(k[0]*a,k[1]),G[2].reshape(k[1], b*c*k[3])],[(-1,1),(1,-2)]).reshape(a, b, c)
			print(G[1].size())
			print(U.size())
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

# # #矩阵化
# # U[3] = U[0].permute(0,1,2).reshape(a,b*c)
# # U[4] = U[0].permute(1,0,2).reshape(b,a*c)
# # U[2] = U[0].permute(2,0,1).reshape(c,a*b)
# # U[1] = U[0].permute(0,1,2).reshape(a*b,c)


# # U_node=[0 for x in range(0,5)]
# # #U_node[0] = tn.Node(U[0].reshape(a,b,c,1))
# # U_node[0] = U[0].reshape(a,b,c,1)
# # #svd分解部分
# # for i in range(4,0,-1):
# #     #U_node[i] = tn.Node(U[i])
# #     if i>1:
# #     	x_mat = torch.matmul(U[i],U[i].T)
# #     	_,U_=torch.eig(x_mat,True)
# #     	U_node[i]=U_[:,:k[i]]
# #     else:
# #     	#U_node[i],_,_,_=torch.svd(U_node[i], left_edges=[U_node[i][0]], right_edges=[U_node[i][1]],max_singular_values=k[i])
# #     	U_,_,_=torch.svd(U[i])
# #     	U_node[i]=U_[:,:k[i]]
# # #TTM
# # #U_node[1].tensor = U_node[1].tensor.reshape(a,b,r)
# # U_node[1] = U_node[1].reshape(a,b,r)
# # B[1] = tn.ncon([U_node[1],U_node[3],U_node[4]],[(1,2,-3),(1,-1),(2,-2)])
# # B[0] = tn.ncon([U_node[0],U_node[1],U_node[2]],[(1,2,4,-3),(1,2,-1),(4,-2)])

# if(calRecover):
# #还原原始tensor
# 	U1 = tn.ncon([B[1],U_node[3],U_node[4]],[(1,2,-3),(-1,1),(-2,2)])
# 	U0_r = tn.ncon([B[0],U1,U_node[2]],[(1,2,-4),(-1,-2,1),(-3,2)])

# 	#U0_r = torch.squeeze(U0_r.tensor)
# 	U0_r = torch.squeeze(U0_r)
# 	err=torch.norm(U[0] - U0_r)/torch.norm(U[0])

# 	print('err is ',err)
	
# end_time = time()
# run_time = end_time-begin_time
# print ('cost time is：',run_time)