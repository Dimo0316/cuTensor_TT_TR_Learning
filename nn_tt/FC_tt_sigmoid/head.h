// #pragma once
#ifndef _SETTING_H_
#define _SETTING_H_

#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>
#include <cufft.h>
#include <cuda_fp16.h>

#define IPNNUM 784
#define HDNNUM 1024
#define OPNNUM 10
#define BATCHSIZE 100
#define FLOATZERO 0.000001
typedef float dt;

using namespace std;
__global__ void Vector2_Multiply_By_Elements (const dt* a, const dt* b, int n, dt* out);
__global__ void Vector1_Multiply_By_Elements (const dt* a, dt* b, int n);
__global__ void activate_sigmoid(dt *d_A,int b,int a);
__global__ void activate_relu(dt *d_A,int a);
__global__ void activate_softmax_exp(dt *d_A,int a);
__global__ void activate_softmax_sum(dt *d_sum, dt *d_A, int batch);
__global__ void activate_softmax_final(dt *d_A, dt *d_Asum, int a);
__global__ void loss_gpu_cross_kernel(dt *d_temp, dt *out, dt *traget, int number);

void warmup();
void printTensor(float *d_des,long m,long n,long l);
void activate_softmax(dt *d_A, int out, int batch);
dt loss_gpu_cross(dt *A,dt *B,int n,int batch,cublasHandle_t handle);


void forward_cuda(dt *input, dt *G1, dt *outG1, dt *G2,
                  dt *outG2, dt *G3, dt *outG3, dt *G4,
                  dt *outG4, dt *W2, dt* outFinal, dt b1,dt b2,
                  int* inp_modes, int* out_modes, int* ranks, int* out_values, int out, int batch, cublasHandle_t handle);
void back_cuda(dt *Y,dt *Y_hat, dt *W2,
               dt *outG4, dt *G4, dt *outG3, dt *G3,
               dt *outG2, dt *G2, dt *outG1, dt *G1, dt *input, 
               int* inp_modes, int* out_modes, int* ranks, int* out_values, int out, int batch, dt rate, cublasHandle_t handle);

#endif /* SETTING_H_ */