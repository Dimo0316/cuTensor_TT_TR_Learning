# cuTensor_TT_TR_Learning
This repository contains applications for TT tensor learning using GPU tensor cores, including TT/TR tensor decompostion, TT tensor layer and DMRG algorithm. 

## File structure

> cuTensor_TT_TR_Learning
>> ttdecomposition
>>> third_order -------- third order tt decomposition <br>
>>>> python -------- python implementations  <br>
>>>> unopt -------- unoptimized on single GPU <br>
>>>> opt -------- optimized on single GPU <br>

>>> high_order -------- high order tt decomposition <br>
>>>> python -------- python implementations <br>
>>>> ttdec6_singleGPU.cu -------- optimized single GPU <br>
>>>> ttdec6_multiGPUs.cu -------- optimized on multiple GPUs <br>


>> trdecomposition
>>> third_order -------- third order tr decomposition <br>
>>>> python -------- python implementations  <br>
>>>> unopt -------- unoptimized on single GPU <br>
>>>> opt -------- optimized on single GPU <br>

>>> high_order -------- high order tr decomposition <br>
>>>> python -------- python implementations <br>
>>>> trdec6_singleGPU.cu -------- optimized on single GPU  <br>
>>>> trdec6_multiGPUs.cu -------- optimized on multiple GPUs  <br>


>> TT_tensor_layer
>>> mnist.py -------- fully connect <br>
>>> mnist_tt.py -------- TT tensor layer <br>
>>> mnist_tt_half.py -------- TT tensor layer using apex <br>

>> DMRG
>>> python -------- TensorNetwork-JAX <br>
>>> baseline ------- unoptimized <br>
>>> optimized -------- optimized using GPU tensor cores<br>
