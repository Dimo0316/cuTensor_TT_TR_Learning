/*************************************************************************
	> File Name: func.h
	> Author: honghao
	> Mail: honghao@shu.edu.cn 
	> Created Time: 2020年04月2日 星期四 14时25分17秒
 ************************************************************************/
#include "head.h"

__global__  void floattohalf(dt *AA,half *BB,long m){
    long i = blockIdx.x*blockDim.x+threadIdx.x;
    const long temp = blockDim.x*gridDim.x;
    if(i<m){
        BB[i] = __float2half(AA[i]);
        i+=temp;
    }
    __syncthreads();
}


__global__ void matrixGetSub_U(dt *d_src, dt *d_dst, int m, int ks, int k){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    const long temp = blockDim.x*gridDim.x;
    if(i<m*k){
        d_dst[i]=d_src[i];
        // d_Ainv[i] = d_A[];
        i+=temp;
    }
    __syncthreads();
}

__global__ void matrixGetSub_VT(dt *d_src, dt *d_dst, long ks, long n, long k){
    long i = blockIdx.x*blockDim.x+threadIdx.x;
    const long temp = blockDim.x*gridDim.x;
    if(i<k*n){
        int row = i%k;
        int col = i/k;
        d_dst[i]=d_src[row+col*ks];
        i+=temp;
    }
    __syncthreads();
}

void matrixGetSub_VT_host(dt *d_src, dt *d_dst, long ks, long n, long k){
    dim3 threads(1024,1,1);
    dim3 blocks((n*k+1024-1)/1024,1,1);   
    matrixGetSub_VT<<<blocks,threads>>>(d_src,d_dst,ks,n,k);
}


__global__ void matrixInvertColumn(dt *d_A, dt *d_Ainv, int m, int n){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<m*n){
        int row = i%m;
        int col = i/m;
        d_Ainv[row+(n-col-1)*m]=d_A[i];
        // d_Ainv[i] = d_A[];
    }
    __syncthreads();
}


void f2h(dt *A,half *B,long num){
    dim3 threads(1024,1,1);
    dim3 blocks((num+1024-1)/1024,1,1);   
    floattohalf<<<blocks,threads>>>(A,B,num);
}

void genTTTensor(dt *T,long a,long b,long c,long r){
    dt *AA,*BB,*CC;
    cudaHostAlloc((void**)&AA,sizeof(dt)*a*r,0);
    cudaHostAlloc((void**)&BB,sizeof(dt)*b*r,0);
    cudaHostAlloc((void**)&CC,sizeof(dt)*c*r,0);
    for(long i = 0;i<a*r;i++){
        AA[i]=rand()*0.1/(RAND_MAX*0.1)*2.0 - 1.0;
    }
    for(long i = 0;i<b*r;i++){
        BB[i]=rand()*0.1/(RAND_MAX*0.1)*2.0 - 1.0;
    }
    for(long i = 0;i<c*r;i++){
        CC[i]=rand()*0.1/(RAND_MAX*0.1)*2.0 - 1.0;
    }
    dt *d_T,*d_CC,*d_BB,*d_AA;
    cudaMalloc((void**)&d_AA,sizeof(dt)*a*r);
    cudaMalloc((void**)&d_BB,sizeof(dt)*b*r);
    cudaMalloc((void**)&d_CC,sizeof(dt)*c*r);
    cudaMalloc((void**)&d_T,sizeof(dt)*a*b*c);
    cudaMemcpyAsync(d_BB,BB,sizeof(dt)*b*r,cudaMemcpyHostToDevice,0);
    cudaMemcpyAsync(d_CC,CC,sizeof(dt)*c*r,cudaMemcpyHostToDevice,0);
    cudaMemcpyAsync(d_AA,AA,sizeof(dt)*a*r,cudaMemcpyHostToDevice,0);
    dt *d_CKRB;
    cudaMalloc((void**)&d_CKRB,sizeof(dt)*c*r*b);
    cudaDeviceSynchronize();

    // printMatrix_Device(a,r,d_AA,a,"AA");
    // printMatrix_Device(b,r,d_BB,b,"BB");
    // printMatrix_Device(c,r,d_CC,c,"CC");

    //X1=A*(CkrB)'  a*r  r*(bc)
    dt alpha = 1.0;
    dt beta = 0.0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,
    		b,c,1,
    		&alpha,
    		d_BB,b,b,
    		d_CC,c,c,
    		&beta,d_CKRB,
    		b,b*c,r);
    // printMatrix_Device(b,c*r,d_CKRB,b,"CkrB");

    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,b*c,r,&alpha,d_AA,a,d_CKRB,b*c,&beta,d_T,a);
    // printMatrix_Device(a,b*c,d_T,a,"T");

    cudaMemcpyAsync(T,d_T,sizeof(dt)*a*b*c,cudaMemcpyDeviceToHost,0);
    cudaDeviceSynchronize();

    cudaFree(d_AA);
    cudaFree(d_BB);
    cudaFree(d_CC);
    cudaFree(d_CKRB);
    cudaFree(d_T);
    cudaFreeHost(AA);
    cudaFreeHost(BB);
    cudaFreeHost(CC);
    cublasDestroy(handle);
}

__global__ void warmup()
{
    int sum=0;
    for(int i = 0; i < 1000; i++) {
        sum+=i;
    }
}   

void warmupcu(){
    warmup<<<1,1>>>();
}

void printMatrix(int m, int n, const dt*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            float Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        }
    }
    cout << " ------------------------------------" << endl;
}

void printMatrix_Device(int m, int n, const dt*d_A, int lda, const char* name)
{
    dt *h_A = NULL;
    cudaHostAlloc((void**)&h_A , sizeof(dt)*m*n, 1);
    cudaMemcpy(h_A, d_A, sizeof(dt)*m*n, cudaMemcpyDeviceToHost);
    printMatrix(m, n, h_A, lda, name);
    if(h_A) cudaFreeHost(h_A);
}

__global__ void sub(dt *A,dt *B,long a,long b,long c)
{
    long long i = blockIdx.x*blockDim.x+threadIdx.x;
    const long long temp = blockDim.x*gridDim.x;
    while(i<a*b*c)
    {
        B[i] = A[i] - B[i];
        i+=temp;
    }
    __syncthreads();
}

