#ifndef GUARD_func_h
#define GUARD_func_h
typedef float dt;

void doDMRG_MPO(dt *M,dt **d_A,dt *ML,dt *MR,int (*sizeA)[3],int numsweep,int dispon,bool updateon,int maxit,int krydim,int hh);
__global__ void transpose(float *A,float* AT,int m,int n);
__global__ void upper(float *A,float *R,int m,int n);
void QR(dt *d_A,dt *d_R,int m,int n,cusolverDnHandle_t cusolverH);
void RQ(dt *d_A,dt *d_Q,dt *d_QRweight,int a,int b,int c,cublasHandle_t handle,cusolverDnHandle_t cusolverH);
float norm(dt *d_l,dt sh,int len,cublasHandle_t handle);
void dgmm(dt *d_S,dt *d_VH,dt *d_svh,int m,int n,cublasHandle_t handle);
void matmul(dt *d_r,dt *d_A,int m,int d,int n,cublasHandle_t handle);
void division(dt *d_A,dt norm_s,dt *d_r,int l,int d, int r,cublasHandle_t handle);
__global__ void together(float *d_A,float *d_des,int a,int b,int c);
__global__ void mode2(dt *A,dt *B,long m,long n,long r);
__global__ void initIdeMat(float *AA,int m);
__global__ void Init_P(float *d_P,int m);
void ncon(dt *d_L,dt *d_M,dt *d_A,int *sizeA,int *sizeL,dt *d_out,cublasHandle_t handle,cutensorHandle_t tensor_handle);
void permute(dt *d_L,int a,int b,int c,int d,int e,cutensorHandle_t tensor_handle,int flag);
float lanczos(dt *d_psiGround,dt *d_L,dt *d_M,dt *d_R,int maxit,int krydim,int size_psi,int *sizeL,int *sizeA,int *sizeR,cublasHandle_t handle,cusolverDnHandle_t cusolverH,cutensorHandle_t tensor_handle);
void per_lmr(dt *d_L,int a,int b,int c,int d,int e,int f);
float cuda_dot(dt *A,dt *B,int n,dt result,cublasHandle_t handle);
void eig(dt *d_A,dt *d_W,int n);
void ncon2(dt *d_R,dt *d_M,dt *d_A,int *sizeA,int *sizeR,dt *d_out,cublasHandle_t handle,cutensorHandle_t tensor_handle);
__global__ void tensorToMode3(dt *T1,dt *T2,int m,int n,int k);
__global__ void tensorToMode1(dt *T1,dt *T2,int m,int n,int k );
#endif