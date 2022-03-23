import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import time
import os

# os.environ["CUDA_VISIBLE_DEVICES"]

def genTRTensor(G, k, i, datatype):
    np.random.seed(1)
    G[6] = np.random.rand(k[5],i*k[6])
    G[5] = np.random.rand(k[4]*i,k[5])
    G[4] = np.random.rand(k[3]*i,k[4])
    G[3] = np.random.rand(k[2]*i,k[3])
    G[2] = np.random.rand(k[1]*i,k[2])
    G[1] = np.random.rand(i,k[0]*k[1])  # k[0]* n1 *k[1]  ->  n1 * k[0] *k[1] 

    G[5] = np.matmul(G[5],G[6]).reshape(k[4],i*i*k[6])
    G[4] = np.matmul(G[4],G[5]).reshape(k[3],i*i*i*k[6])
    G[3] = np.matmul(G[3],G[4]).reshape(k[2],i*i*i*i*k[6])
    G[2] = np.matmul(G[2],G[3]).reshape(k[1],i*i*i*i*i*k[6]) # k[1] * n2*n3*n4*n5*n6 *k[0] ->  k[0] * k[1] *n2*n3*n4*n5*n6
    G[2] = tf.transpose(tf.reshape(G[2],(k[1],i*i*i*i*i,k[6])),(2,0,1))
    G[2] = tf.reshape(G[2], (k[6]*k[1], i*i*i*i*i))
    X = np.matmul(G[1],G[2]).reshape(i,i,i,i,i,i).astype(datatype)
    return X

def reCover(G, k, i):
    G[5] = tf.reshape(np.matmul(tf.reshape(G[5], (k[4]*i,k[5])) , tf.reshape(G[6], (k[5],i*k[6]))), (k[4],i*i*k[6]))
    G[4] = tf.reshape(np.matmul(tf.reshape(G[4], (k[3]*i,k[4])), G[5]), (k[3],i*i*i*k[6]))
    G[3] = tf.reshape(np.matmul(tf.reshape(G[3], (k[2]*i,k[3])), G[4]), (k[2],i*i*i*i*k[6]))
    G[2] = tf.reshape(np.matmul(tf.reshape(G[2], (k[1]*i,k[2])), G[3]), (k[1],i*i*i*i*i*k[6])) # k[1] * n2*n3*n4*n5*n6 *k[0] ->  k[0] * k[1] *n2*n3*n4*n5*n6
    G[2] = tf.reshape(tf.transpose(tf.reshape(G[2],(k[1], i*i*i*i*i ,k[6])),(2,0,1)), (k[6]*k[1], i*i*i*i*i))
    G[1] = tf.reshape(tf.transpose(tf.reshape(G[1],(k[0], i ,k[1])), (1,0,2)), (i, k[0]*k[1]))

    X = tf.reshape(np.matmul(G[1],G[2]), (i,i,i,i,i,i))
    return X

def TRdecomposition(tf_tens, G, k, i):
#获得G[1]
    tens = tf.reshape(tf_tens,(i,i*i*i*i*i))
    s, u, v = tf.linalg.svd(tens, full_matrices=False)
    G[1] = tf.transpose(tf.reshape(u[:, 0:k[0]*k[1]],(i,k[0],k[1])), (1,0,2))
    s = s[0:(k[0]*k[1])]
    v = v[:, 0:(k[0]*k[1])]
    tens = tf.matmul(tf.linalg.diag(s), tf.transpose(v,(1,0)))
    tens = tf.transpose(tf.reshape(tens, (k[0],k[1],i*i*i*i*i)), (1,2,0)) #k[1] , i*i*i*i*i, k[0]
    tens = tf.reshape(tens, (k[1]*i, i*i*i*i*k[6]))

#获得G[2]
    s, u, v = tf.linalg.svd(tens, full_matrices=False)
    G[2] = tf.reshape(u[:, 0:k[2]],(k[1],i,k[2]))
    s = s[0:k[2]]
    v = v[:, 0:k[2]]
    tens = tf.matmul(tf.linalg.diag(s), tf.transpose(v,(1,0)))
    tens = tf.reshape(tens, (k[2]*i, i*i*i*k[6]))

#获得G[3]
    s, u, v = tf.linalg.svd(tens, full_matrices=False)
    G[3] = tf.reshape(u[:, 0:k[3]],(k[2],i,k[3]))
    s = s[0:k[3]]
    v = v[:, 0:k[3]]
    tens = tf.matmul(tf.linalg.diag(s), tf.transpose(v,(1,0)))
    tens = tf.reshape(tens, (k[3]*i, i*i*k[6]))

#获得G[4]
    s, u, v = tf.linalg.svd(tens, full_matrices=False)
    G[4] = tf.reshape(u[:, 0:k[4]],(k[3],i,k[4]))
    s = s[0:k[4]]
    v = v[:, 0:k[4]]
    tens = tf.matmul(tf.linalg.diag(s), tf.transpose(v,(1,0)))
    tens = tf.reshape(tens, (k[4]*i, i*k[6]))

#获得G[5] G[6]
    s, u, v = tf.linalg.svd(tens, full_matrices=False)
    G[5] = tf.reshape(u[:, 0:k[5]],(k[4],i,k[5]))
    s = s[0:k[5]]
    v = v[:, 0:k[5]]
    tens = tf.matmul(tf.linalg.diag(s), tf.transpose(v,(1,0)))
    G[6] = tf.reshape(tens, (k[5], i, k[6]))

class _DecompositionsTest():

  def testTRTensor(self):
    calTimes = 5
    writeFile = False
    calError = True

    for i in range(8, 36, 4):
      print(i)
      shape = (i, i, i, i, i, i)
      r = 4
      k = [2, 2, r, r, r, r, 2]
      if(i>16):
        r = 8
        k = [2, 4, r, r, r, r, 2]

      np.random.seed(1)
      G=[0 for x in range(0,7)]

      datatype = self.dtype.as_numpy_dtype
      X = genTRTensor(G, k, i, datatype)
      print(X.shape)
      print(k)

      time_start_initial = time.time()
      tf_tens = tf.constant(X)
      time_end_initial = time.time()

      TRdecomposition(tf_tens, G, k, i)

      time_start = time.time()

      for j in range(0, calTimes):
        TRdecomposition(tf_tens, G, k, i)

      time_end = time.time()
      # print('time cost ', (time_end - time_start) / calTimes, 's')
      print('time cost ', (time_end - time_start) / calTimes + time_end_initial - time_start_initial, 's')

      if(writeFile):
        f = open('t3f.txt', 'a')
        f.write(str(i))
        f.write('   ')
        f.write(str((time_end - time_start) / calTimes + time_end_initial - time_start_initial))
        f.write('\n')
        f.close()

      if(calError):
        G[0] = reCover(G, k, i)
        err = tf.linalg.norm(tf_tens-G[0])/tf.linalg.norm(tf_tens)
        print("err:",err)


class DecompositionsTestFloat32(tf.test.TestCase, _DecompositionsTest):
  dtype = tf.float32



if __name__ == "__main__":
  tf.test.main()
