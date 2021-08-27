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
float trdec_gpu_float(float* h_tensor, int* h_n_array, int* h_r_array,double *eps, float* singleError, float* singleComRatio);
void trals_gpu_float(float* h_tensor, int* h_n_array, int* h_r_array, double* eps, int maxiter);
void genTRTensor(dt *T,long a,long b,long c,long r);
void printTensor(const dt *d_des,long m,long n,long l, const char* name);
__global__ void hardm(dt *M,dt *N,dt *res,long  m);
__global__ void tensorToMode2(dt *T1,dt *T2,int m,int n,int k);
__global__ void tensorToMode3(dt *T1,dt *T2,int m,int n,int k);
__global__ void tensorToMode231(dt *T1,dt *T2,int m,int n,int k);
__global__ void elemin(dt *A,dt *B, long n);
__global__  void floattohalf(dt *AA,half *BB,long m);
__global__ void matrixInvertColumn(dt *d_A, dt *d_Ainv, int m, int n);

int pinv(cublasHandle_t cb_handle, cusolverDnHandle_t cs_handle, float * d_X, int n, int p, float * d_Xpinv, cublasGemmAlgo_t algo);

void basicRandSVD(cusolverDnHandle_t cusolverH, cublasHandle_t cublasH, float *d_A, 
            const int m, const int n, const int k, const int s,const int p, cublasGemmAlgo_t algo, float *d_U, float *d_S, float *d_V);
void basicRandSVD_LU(cusolverDnHandle_t cusolverH, cublasHandle_t cublasH, float *d_A, 
    		const int m, const int n, const int k, const int s,const int p, cublasGemmAlgo_t algo, float *d_U, float *d_SVT);
void basicQR(cusolverDnHandle_t cusolverH, float *d_A, int m, int n);
void basicEig(cusolverDnHandle_t cusolverH, cublasHandle_t cublasH, dt *d_A, const int m, dt *d_W);
void f2h(dt *A,half *B,long num);

#endif
