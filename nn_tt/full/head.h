#ifndef SETTING_H_
#define SETTING_H_

#include <bits/stdc++.h>
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<cusolverDn.h>
#include<curand.h>
#include <cufft.h>
#include <cuda_fp16.h>
#include <random>
#define IPNNUM 784
#define HDNNUM 256
//#define H1 
//#define B1
//#define H2
#define OPNNUM 10
#define BATCHSIZE 64

using namespace std;
__global__ void activate(double *d_A,int b,int a);
__global__ void Vector2_Multiply_By_Elements (const double* a, const double* b, int n, double* out);
__global__ void Vector1_Multiply_By_Elements (const double* a, double* b, int n);
void printTensor(float *d_des,long m,long n,long l);
void forward_cuda(double *input,double *W1,double *outh,double *W2,double *outo,double k1,double k2,int in,int hid,int out,int batch,cublasHandle_t handle);
void back_cuda(double *Y,double *Y_hat,double *outh,double *W2,double *input,double *W1,int in,int hid,int out,int batch,double rate,cublasHandle_t handle);
double loss_gpu(double *A,double *B,int n,int batch,cublasHandle_t handle);
void warmup();
#endif /* SETTING_H_ */