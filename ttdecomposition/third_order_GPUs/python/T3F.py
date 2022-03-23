import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import time

from t3f import ops
import t3f
from t3f import shapes
from t3f import decompositions
from t3f import initializers
import tensornetwork as tnk



class _DecompositionsTest():

  def testTTTensor(self):
    calTimes = 10
    writeFile = False
    calError = True
    tnk.set_default_backend("pytorch")
    for i in range(100, 200, 100):
      print(i)
      shape = (i, i, i)
      r = 56
      if(i>600):
        r = i//10//8*8

      k = [1, r, r, 1]
      # rank_single = i // 10 // 8 * 8

      np.random.seed(1)
      G=[0 for x in range(0,4)]
      G[3] = np.random.rand(k[2],i*k[3])
      G[2] = np.random.rand(k[1]*i,k[2])
      G[1] = np.random.rand(k[0]*i,k[1])
      # ten = t3f.random_tensor(shape, tt_rank=k)
      # print(t3f.shape(ten))
      G[2] = np.matmul(G[2],G[3]).reshape(k[1],i*i*k[3])
      
      X = np.matmul(G[1],G[2]).reshape(i,i,i).astype(self.dtype.as_numpy_dtype)
      # print(X.shape)
            # G[2] = tnk.ncon([G[2],G[3]],[(-1,-2,1),(1,-3,-4)]).reshape(r, i*i, 1)
      # X = tnk.ncon([G[1],G[2]],[(-1,-2,1),(1,-3,-4)]).reshape(i,i,i)
      # tens = np.random.rand(*shape).astype(self.dtype.as_numpy_dtype)
      tf_tens = tf.constant(X)
      print(tf_tens)
    
      # tt_tens = decompositions.to_tt_tensor(tf_tens, max_tt_rank=r)

      time_start = time.time()
      for j in range(0, calTimes):
        tt_tens = decompositions.to_tt_tensor(tf_tens, max_tt_rank=r)
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
        # t3f.full(tt_tens)
        err = tf.linalg.norm(tf_tens-t3f.full(tt_tens))/tf.linalg.norm(tf_tens)
        print("err:",err)
      # self.assertAllClose(tens, self.evaluate(ops.full(tt_tens)))
      # dynamic_tt_ranks = self.evaluate(shapes.tt_ranks(tt_tens))
      # static_tt_ranks = tt_tens.get_tt_ranks().as_list()
      # self.assertAllEqual(dynamic_tt_ranks, static_tt_ranks)


class DecompositionsTestFloat32(tf.test.TestCase, _DecompositionsTest):
  dtype = tf.float32



if __name__ == "__main__":
  tf.test.main()
