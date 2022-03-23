/*************************************************************************
	> File Name: func.h
	> Author: honghao
	> Mail: honghao@shu.edu.cn 
	> Created Time: 2020年04月2日 星期四 14时25分17秒
 ************************************************************************/
#ifndef GUARD_func_h
#define GUARD_func_h

typedef float dt;


void warmupcu();
void printMatrix(int m, int n, const dt*A, int lda, const char* name);
void printMatrix_Device(int m, int n, const dt*d_A, int lda, const char* name);
float ttdec_gpu_float(float* h_tensor, int* h_n_array, int* h_r_array, double* eps, float* N, float* M);
void genTTTensor(dt *T,long a,long b,long c,long r);
void basicRandSVD(cusolverDnHandle_t cusolverH, cublasHandle_t cublasH, float *d_A, 
            const int m, const int n, const int k, const int s,const int p, cublasGemmAlgo_t algo, float *d_U, float *d_S, float *d_V);
void basicRandSVD_LU(cusolverDnHandle_t cusolverH, cublasHandle_t cublasH, float *d_A, 
    const int m, const int n, const int k, const int s,const int p, cublasGemmAlgo_t algo, float *d_U, float *d_SVT);
void basicQR(cusolverDnHandle_t cusolverH, float *d_A, int m, int n);
void basicEig(cusolverDnHandle_t cusolverH, cublasHandle_t cublasH, dt *d_A, const int m, dt *d_W);
double norm2HH(float *A, long long len);
void f2h(dt *A,half *B,long num);
void matrixGetSub_VT_host(dt *d_src, dt *d_dst, long ks, long n, long k);

__global__ void sub(dt *A,dt *B,long a,long b,long c);
__global__  void floattohalf(dt *AA,half *BB,long m);
__global__ void matrixInvertColumn(dt *d_A, dt *d_Ainv, int m, int n);

#endif
