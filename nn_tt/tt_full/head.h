#ifndef SETTING_H_
#define SETTING_H_

#include <bits/stdc++.h>
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<cusolverDn.h>
#include<curand.h>
#include <assert.h>
#include <cufft.h>
#include <cuda_fp16.h>
#define IPNNUM 784
#define HDNNUM 256
#define K1 64
#define K2 64
//#define H2 128
#define OPNNUM 10
#define BATCHSIZE 64

using namespace std;
__global__ void upper1(double *A,double *R,int m,int n);
__global__ void activate(double *d_A,int b,int a);
__global__ void Vector2_Multiply_By_Elements (const double* a, const double* b, int n, double* out);
__global__ void Vector1_Multiply_By_Elements (const double* a, double* b, int n);
void printTensor(float *d_des,long m,long n,long l);
void forward_cuda(double *input,double *W_U2,double *outu2,double *W_B1,
                  double *outB1,double *W_U1,double *outU1,double *W2,double *outo,
                  double b1,double b2,
                  int in,int k2,int k1,int hid,int out,int batch,cublasHandle_t handle);
void back_cuda(double *Y,double *Y_hat,
               double *outU1,double *W2,
               double *outB1,double *W_U1,
               double *outu2,double *W_B1,
               double *input,double *W_U2,int in,int k2,int k1,int hid,int out,int batch,double rate,cublasHandle_t handle);
double loss_gpu(double *A,double *B,int n,int batch,cublasHandle_t handle);
void warmup();
void tt(double *d_A, double *G1, double *G2,double *G3, int m, int n, int k1, int k2, cublasHandle_t handle);
void tt_random(double *d_A,double *G1, double *G2,double *G3, int m, int n, int k1, int k2, cublasHandle_t handle);
void eig(double *d_A,int m,int n,cusolverDnHandle_t cusolverH);
#endif /* SETTING_H_ */