import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import time
import os
from t3f import ops
import t3f
from t3f import shapes
from t3f import decompositions
from t3f import initializers
# import tensornetwork as tnk

# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
def genTTensor(k, i, datatype):
    np.random.seed(1)
    G=[0 for x in range(0,7)]
    G[6] = np.random.rand(k[5],i*k[6])
    G[5] = np.random.rand(k[4]*i,k[5])
    G[4] = np.random.rand(k[3]*i,k[4])
    G[3] = np.random.rand(k[2]*i,k[3])
    G[2] = np.random.rand(k[1]*i,k[2])
    G[1] = np.random.rand(k[0]*i,k[1])

    G[5] = np.matmul(G[5],G[6]).reshape(k[4],i*i*k[6])
    G[4] = np.matmul(G[4],G[5]).reshape(k[3],i*i*i*k[6])
    G[3] = np.matmul(G[3],G[4]).reshape(k[2],i*i*i*i*k[6])
    G[2] = np.matmul(G[2],G[3]).reshape(k[1],i*i*i*i*i*k[6])
    
    X = np.matmul(G[1],G[2]).reshape(i,i,i,i,i,i).astype(datatype)

    return X

class _DecompositionsTest():
  
  def testTTTensor(self):
    calTimes = 10
    writeFile = False
    calError = False
    # tnk.set_default_backend("pytorch")
    for i in range(8, 36, 4):
      print(i)
      shape = (i, i, i, i, i, i)
      r = 4
      if(i>24):
        r = 8
      ranks = [1, r, r, r, r, r, 1]
      # rank_single = i // 10 // 8 * 8

      
      X = genTTensor(ranks, i, self.dtype.as_numpy_dtype)
      # print(X.shape)
            # G[2] = tnk.ncon([G[2],G[3]],[(-1,-2,1),(1,-3,-4)]).reshape(r, i*i, 1)
      # X = tnk.ncon([G[1],G[2]],[(-1,-2,1),(1,-3,-4)]).reshape(i,i,i)
      # tens = np.random.rand(*shape).astype(self.dtype.as_numpy_dtype)
      # print(X.device)
      time_start_initial = time.time()
      tf_tens = tf.constant(X)
      time_end_initial = time.time()

      # print("shuju zai  zheli", tf_tens.device)
      
      # tt_tens = decompositions.to_tt_tensor(tf_tens, max_tt_rank=r)
      tt_tens = decompositions.to_tt_tensor(tf_tens, max_tt_rank=r)

      # print("shuju zai  zheli", tf_tens.device)
      time_start = time.time()
      for j in range(0, calTimes):
        tt_tens = decompositions.to_tt_tensor(tf_tens, max_tt_rank=r)
      time_end = time.time()
      print('time cost ', (time_end - time_start) / calTimes + time_end_initial - time_start_initial, 's')


      if(writeFile):
        f = open('t3f.txt', 'a')
        f.write(str(i))
        f.write('   ')
        f.write(str((time_end - time_start) / calTimes))
        f.write('\n')
        f.close()
      if(calError):
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
