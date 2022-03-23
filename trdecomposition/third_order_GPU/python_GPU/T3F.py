import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import time


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

class _DecompositionsTest():

  def testTRTensor(self):
    calTimes = 5
    writeFile = True
    calError = False

    for i in range(100, 1200, 100):
      print(i)
      shape = (i, i, i)
      r = 56
      if(i>600):
        r = i//10//8*8

      k = [2, 28, 100, 2]
      # rank_single = i // 10 // 8 * 8
      m = 50
      np.random.seed(1)
      G=[0 for x in range(0,4)]
      # G[2] = np.transpose(np.matmul(G[2],G[3]).reshape(k[1],i*i,k[3]),((0,2,1))).reshape(k[1]*k[3],i*i)
      
      # X = np.matmul(G[1],G[2]).reshape(i,i,i).astype(self.dtype.as_numpy_dtype)
      

      G[1] = np.random.rand(i, m)
      G[2] = np.random.rand(i, m)
      G[3] = np.random.rand(m, i)
      # print(G[1].shape)
      one = np.ones((i,1))
      G[1] = np.kron(G[1],one)
      G[2] = repmat(G[2],i,1)
      G[2] = G[1]*G[2]
      X = np.matmul(G[2],G[3]).reshape(i,i,i).astype(self.dtype.as_numpy_dtype)
      # print(X.shape)
      tf_tens = tf.constant(X)
      # print(tf_tens)
      
      if i==100:
        G[0] = tf.reshape(tf_tens,(i,i*i))
        s, u, v = tf.linalg.svd(G[0], full_matrices=False)
        G[1] = tf.transpose(tf.reshape(u[:, 0:k[0]*k[1]],(i,k[0],k[1])), (1,0,2))
        s = s[0:(k[0]*k[1])]
        v = v[:, 0:(k[0]*k[1])]
        # tt_cores.append(u)
        # print(v)
        G[0] = tf.matmul(tf.linalg.diag(s), tf.transpose(v,(1,0)))
        tens = tf.transpose(tf.reshape(G[0], (k[0],k[1],i*i)), (1,2,0))
        tens = tf.reshape(tens, (k[1]*i,i*k[3]))
        s, u, v = tf.linalg.svd(tens, full_matrices=False)
        # print(u.shape)
        # print(s.shape)
        # print(v.shape)
        G[2] = tf.reshape(u[:, 0:k[2]],(k[1],i,k[2]))
        # print(G[2].shape)
        # tt_cores.append(u)
        s = s[0:k[2]]
        v = v[:, 0:k[2]]
        G[3] = tf.matmul(tf.linalg.diag(s), tf.transpose(v))
        G[3] = tf.reshape(G[3], (k[2],i,k[3]))

      time_start = time.time()
      for j in range(0, calTimes):
        G[0] = tf.reshape(tf_tens,(i,i*i))
        s, u, v = tf.linalg.svd(G[0], full_matrices=False)
        G[1] = tf.transpose(tf.reshape(u[:, 0:k[0]*k[1]],(i,k[0],k[1])), (1,0,2))
        s = s[0:(k[0]*k[1])]
        v = v[:, 0:(k[0]*k[1])]
        # tt_cores.append(u)
        # print(v)
        G[0] = tf.matmul(tf.linalg.diag(s), tf.transpose(v,(1,0)))
        tens = tf.transpose(tf.reshape(G[0], (k[0],k[1],i*i)), (1,2,0))
        tens = tf.reshape(tens, (k[1]*i,i*k[3]))
        s, u, v = tf.linalg.svd(tens, full_matrices=False)
        # print(u.shape)
        # print(s.shape)
        # print(v.shape)
        G[2] = tf.reshape(u[:, 0:k[2]],(k[1],i,k[2]))
        # print(G[2].shape)
        # tt_cores.append(u)
        s = s[0:k[2]]
        v = v[:, 0:k[2]]
        G[3] = tf.matmul(tf.linalg.diag(s), tf.transpose(v))
        G[3] = tf.reshape(G[3], (k[2],i,k[3]))
      time_end = time.time()
      print('time cost ', (time_end - time_start) / calTimes, 's')

      if(writeFile):
        f = open('t3f.txt', 'a')
        f.write(str(i))
        f.write('   ')
        f.write(str((time_end - time_start) / calTimes))
        f.write('\n')
        f.close()
      if(calError):
        G[3] = tf.reshape(G[3],(k[2],i*k[3]))
        G[2] = tf.reshape(G[2],(k[1]*i,k[2]))
        G[2] = tf.transpose(tf.reshape(tf.matmul(G[2],G[3]),(k[1],i*i,k[3])),(2,0,1))
        G[2] = tf.reshape(G[2], (k[1]*k[3], i*i))
        G[1] = tf.reshape(tf.transpose(G[1],(1,0,2)),(i, k[0]*k[1]))
        G[0] = tf.reshape(tf.matmul(G[1],G[2]),(i,i,i))
        err = tf.linalg.norm(tf_tens-G[0])/tf.linalg.norm(tf_tens)
        print("err:",err)


class DecompositionsTestFloat32(tf.test.TestCase, _DecompositionsTest):
  dtype = tf.float32



if __name__ == "__main__":
  tf.test.main()
