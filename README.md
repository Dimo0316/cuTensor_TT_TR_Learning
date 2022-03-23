#                     TT and TR Tensor Learning Using GPU Tensor Cores

This repository contains applications for TT tensor learning using GPU tensor cores, including TT/TR tensor decompostion, TT tensor layer and DMRG algorithm. 


## File structure

> cuTensor_TT_TR_Learning
>> ttdecomposition
>>> third_order_GPU -------- third order tt decomposition <br>
>>>> python -------- python implementations  <br>
>>>> unopt -------- unoptimized on single GPU <br>
>>>> opt -------- optimized on single GPU <br>


>>> high_order_GPUs -------- high order tt decomposition <br>
>>>> python -------- python implementations <br>
>>>> ttdec6_singleGPU.cu -------- optimized single GPU <br>
>>>> ttdec6_multiGPUs_2.cu -------- optimized on two GPUs <br>
>>>> ttdec6_multiGPUs_4.cu -------- optimized on four GPUs <br>
>>>> ttdec6_multiGPUs_8.cu -------- optimized on eight GPUs <br>


>> trdecomposition
>>> third_order_GPU -------- third order tr decomposition <br>
>>>> python -------- python implementations  <br>
>>>> unopt -------- unoptimized on single GPU <br>
>>>> opt -------- optimized on single GPU <br>

>>> high_order_GPUs -------- high order tr decomposition <br>
>>>> python -------- python implementations <br>
>>>> trdec6_singleGPU.cu -------- optimized on single GPU  <br>
>>>> trdec6_multiGPUs_2.cu -------- optimized on two GPUs  <br>
>>>> trdec6_multiGPUs_4.cu -------- optimized on four GPUs  <br>
>>>> trdec6_multiGPUs_8.cu -------- optimized on eight GPUs  <br>

>> ablation -------- ablition experiments <br>
>>> edrsvd_GPU -------- only EDrSVD <br>
>>> edrsvd_pipeline_GPU -------- EDrSVD using piepline <br>
>>> edrsvd_rensor_contractions_GPU -------- EDrSVD using tensor contractions <br>

>> TT_tensor_layer
>>> mnist.py -------- fully connect <br>
>>> mnist_tt.py -------- TT tensor layer <br>
>>> mnist_tt_half.py -------- TT tensor layer using apex <br>

>> DMRG
>>> python -------- TensorNetwork-JAX <br>
>>> baseline ------- unoptimized <br>
>>> optimized -------- optimized <br>


## CONTACT US

> Tensor and Deep Learning Lab LLC. 

> E-Mail:** tensorlet@gmail.com

> github:**[https://github.com/Dimo0316/cuTensor_TT_TR_Learning](https://github.com/Dimo0316/cuTensor_TT_TR_Learning)

> Notice: both US and China patents are filed for this project.