#include <iostream>
#include <fstream>
#include <assert.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <math.h>
#include <curand.h>
#include <cufft.h>
#include <cuda_fp16.h>
#include <cublasXt.h>

// /usr/local/cuda/bin/nvcc -Xcompiler -fopenmp -lcublas -lcusolver -lcurand -std=c++11 ttdec6_multiGPUs2.cu -o multiGPUs2
// /xfs/home/tensor_zy/anaconda3/envs/ht2/bin/nvcc -Xcompiler -fopenmp -lcublas -lcusolver -lcurand -std=c++11 ttdec6_multiGPUs_8.cu -o multiGPUs8

using namespace std;
typedef float dt;   
bool reduceDim =true;
cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP; //CUBLAS_GEMM_DEFAULT_TENSOR_OP CUBLAS_GEMM_DEFAULT
cublasMath_t mathMode = CUBLAS_TENSOR_OP_MATH; //CUBLAS_TENSOR_OP_MATH  CUBLAS_DEFAULT_MATH
cudaError_t cudaStat1 = cudaSuccess;
cudaError_t cudaStat2 = cudaSuccess;
cudaError_t cudaStat3 = cudaSuccess;
cudaError_t cudaStat4 = cudaSuccess;
cudaError_t cudaStat5 = cudaSuccess;
cudaError_t cudaStat6 = cudaSuccess;
const float alpha = 1.0, beta0 = 0.0, beta_1 = -1.0, beta1 = 1.0;

__global__ void warmup()
{
    int sum=0;
    for(int i = 0; i < 1000; i++) {
        sum+=i;
    }
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
// nvcc -lcublas -lcusolver -lcurand -std=c++11 svd.cu -o svd
void printMatrix_Device(int m, int n, const dt*d_A, int lda, const char* name)
{
    dt *h_A = NULL;
    cudaHostAlloc((void**)&h_A , sizeof(dt)*m*n, 1);
    cudaMemcpy(h_A, d_A, sizeof(dt)*m*n, cudaMemcpyDeviceToHost);
    printMatrix(m, n, h_A, lda, name);
    if(h_A) cudaFreeHost(h_A);
}
void warmupcu(){
    warmup<<<1,1>>>();
}

struct GPUTimer
{
    GPUTimer() 
    {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
        cudaEventRecord(start_, 0);
    }

    ~GPUTimer() 
    {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() 
    {
        cudaEventRecord(start_, 0);
    }

    float seconds()
    {
        cudaEventRecord(stop_, 0);
        cudaEventSynchronize(stop_);
        float time;
        cudaEventElapsedTime(&time, start_, stop_);
        return time*1e-3;
    }
    private:
    cudaEvent_t start_, stop_;
};

double norm2HH(float *A, long long len){
  printf("%lld\n", len);
  double norm2Result = 0.0;
  for(long long i = 0; i < len; ++i){
    norm2Result += (A[i] * A[i]);
    // if(i%100000000 == 0)
    //     printf("%f\n", norm2Result);
  }
  norm2Result = sqrtf(norm2Result);
  return norm2Result;
}

void smallargeMatrixMulti_slice_host(cublasHandle_t cublasH, dt *h_A, dt *h_B, const int &ttrank, const int &m,  const long long &n, dt *h_AB, const int &p, const dt beta){
    long long slice = 0;
    if(n%p==0)
        slice = n/p;
    else
        assert("P is not slice");
    
    float *d_tempB = NULL, *d_tempATB = NULL, *d_A=NULL;
    cudaStat1 = cudaMalloc((void**)&d_tempB, sizeof(dt)*m*slice);
    cudaStat2 = cudaMalloc((void**)&d_tempATB, sizeof(dt)*ttrank*slice);
    cudaStat3 = cudaMalloc((void**)&d_A, sizeof(dt)*m*ttrank);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);

    cudaStat3 = cudaMemcpyAsync(d_A, h_A, sizeof(dt)*m*ttrank, cudaMemcpyHostToDevice,0);
    cudaStat4 = cudaDeviceSynchronize();

    // printf("%d\n", slice);
    for(int i=0;i<p;i++){
        if(beta == -1.0){
            // printf("beta: %f\n", beta);
            cudaStat3 = cudaMemcpyAsync(d_tempATB, h_AB+ttrank*slice*i, sizeof(dt)*ttrank*slice, cudaMemcpyHostToDevice,0);
            assert(cudaStat3 == cudaSuccess);
        }
        cudaStat3 = cudaMemcpyAsync(d_tempB, h_B+m*slice*i, sizeof(dt)*m*slice, cudaMemcpyHostToDevice,0);
        cudaStat4 = cudaDeviceSynchronize();
        assert(cudaStat3 == cudaSuccess);
        assert(cudaStat4 == cudaSuccess);
        // printMatrix_Device(m, slice, d_tempB, m, "tempB");
        cublas_status = cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                           ttrank, slice, m,
                           &alpha, d_A, ttrank,
                           d_tempB, m,
                           &beta, d_tempATB, ttrank); 
        assert(cublas_status == CUBLAS_STATUS_SUCCESS);

        cudaStat3 = cudaMemcpyAsync(h_AB+ttrank*slice*i, d_tempATB, sizeof(dt)*ttrank*slice, cudaMemcpyDeviceToHost, 0);
        cudaStat4 = cudaDeviceSynchronize();
        assert(cudaStat3 == cudaSuccess);
        assert(cudaStat4 == cudaSuccess);
    }
    if(d_A     ) cudaFree(d_A); d_A=NULL;
    if(d_tempB  ) cudaFree(d_tempB); d_tempB=NULL;
    if(d_tempATB  ) cudaFree(d_tempATB); d_tempATB=NULL;
}

double calMSE(cublasHandle_t cublasH, dt *h_A, dt *h_G1, dt *h_G2, dt *h_G3, dt *h_G4, dt *h_G5, dt *h_G6, const int *ttRanks, const int *ttDemns, const int p){
    long long calNumber1 = ttDemns[0]*ttDemns[1]*ttDemns[2];
    long long calNumber = calNumber1*(long long)ttDemns[3]*ttDemns[4]*ttDemns[5];
    printf("calNumber : %lld\n", calNumber);
    double normA = norm2HH(h_A, calNumber);
    cout<<"normA: "<<normA<<endl;
    dt *h_G56=NULL, *h_G456=NULL, *h_G3456=NULL, *h_G23456=NULL, *h_G123456=NULL;
    
    cudaStat3 = cudaMallocHost((void**)&h_G56, sizeof(dt)*ttRanks[4] *ttDemns[4]*ttDemns[5]* ttRanks[6]);
    assert(cudaStat3 == cudaSuccess);
    smallargeMatrixMulti_slice_host(cublasH, h_G5, h_G6, ttRanks[4]*ttDemns[4], ttRanks[5], ttDemns[5]*ttRanks[6], h_G56, 1, beta0);

    cudaStat3 = cudaMallocHost((void**)&h_G456, sizeof(dt)*ttRanks[3] *ttDemns[3]*ttDemns[4]*ttDemns[5]* ttRanks[6]);
    assert(cudaStat3 ==cudaSuccess);
    smallargeMatrixMulti_slice_host(cublasH, h_G4, h_G56, ttRanks[3]*ttDemns[3], ttRanks[4], ttDemns[4]*ttDemns[5]*ttRanks[6], h_G456, 1, beta0);
    if(h_G56) cudaFreeHost(h_G56);

    cudaStat3 = cudaMallocHost((void**)&h_G3456, sizeof(dt)*ttRanks[2]*ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5]* ttRanks[6]);
    assert(cudaStat3 == cudaSuccess);
    smallargeMatrixMulti_slice_host(cublasH, h_G3, h_G456, ttRanks[2]*ttDemns[2], ttRanks[3], ttDemns[3]*ttDemns[4]*ttDemns[5]*ttRanks[6], h_G3456, 1, beta0);
    if(h_G456) cudaFreeHost(h_G456);

    cudaStat3 = cudaMallocHost((void**)&h_G23456, sizeof(dt)*ttRanks[1]*ttDemns[1]*ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5]*ttRanks[6]);
    assert(cudaStat3 == cudaSuccess);
    smallargeMatrixMulti_slice_host(cublasH, h_G2, h_G3456, ttRanks[1]*ttDemns[1], ttRanks[2], ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5]*ttRanks[6], h_G23456, 4, beta0);
    if(h_G3456) cudaFreeHost(h_G3456);

    smallargeMatrixMulti_slice_host(cublasH, h_G1, h_G23456, ttRanks[0]*ttDemns[0], ttRanks[1], ttDemns[1]*ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5]*ttRanks[6], h_A, p, beta_1);
    if(h_G23456) cudaFreeHost(h_G23456);

    double normAr = norm2HH(h_A, calNumber);
    cout<<"normAr: "<<normAr<<endl;
    double MSE = normAr/normA;
    return MSE;
}

void generalTTTensor(dt *h_A, const int *ttRanks, const int *ttDemns, const int &p){
    dt *h_G56=NULL, *h_G456=NULL, *h_G3456=NULL, *h_G23456=NULL, *h_G123456=NULL;
    dt *h_G1=NULL,  *h_G2=NULL,  *h_G3=NULL,  *h_G4=NULL,  *h_G5=NULL,  *h_G6=NULL;
    cublasHandle_t cublasH = NULL;
    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    cublas_status = cublasSetMathMode(cublasH, mathMode);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    // srand(123321);
    cudaStat1 = cudaMallocHost((void**)&h_G5, sizeof(dt)*ttRanks[4]*ttDemns[4]*ttRanks[5]);
    cudaStat2 = cudaMallocHost((void**)&h_G6, sizeof(dt)*ttRanks[5]*ttDemns[5]*ttRanks[6]);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    srand(1134);
    for(long long i = 0; i < ttRanks[4]*ttDemns[4]*ttRanks[5]; i++){
        h_G5[i] = (dt) ((rand()*1.0) / (RAND_MAX*1.0))*2.0 - 1.0;
    }
    for(long long i = 0; i < ttRanks[5]*ttDemns[5]*ttRanks[6]; i++){
        h_G6[i] = (dt) ((rand()*1.0) / (RAND_MAX*1.0))*2.0 - 1.0;
    }
    
    cudaStat3 = cudaMallocHost((void**)&h_G56, sizeof(dt)*ttRanks[4] *ttDemns[4]*ttDemns[5]* ttRanks[6]);
    assert(cudaStat3 == cudaSuccess);
    smallargeMatrixMulti_slice_host(cublasH, h_G5, h_G6, ttRanks[4]*ttDemns[4], ttRanks[5], ttDemns[5]*ttRanks[6], h_G56, 1, beta0);
    if(h_G5) cudaFreeHost(h_G5);h_G5=NULL;
    if(h_G6) cudaFreeHost(h_G6);h_G6=NULL;



    cudaStat1 = cudaMallocHost((void**)&h_G4, sizeof(dt)*ttRanks[3]*ttDemns[3]*ttRanks[4]);
    assert(cudaStat1 == cudaSuccess);
    for(long long i = 0; i < ttRanks[3]*ttDemns[3]*ttRanks[4]; i++){
        h_G4[i] = (dt) ((rand()*1.0) / (RAND_MAX*1.0))*2.0 - 1.0;
    }
    cudaStat3 = cudaMallocHost((void**)&h_G456, sizeof(dt)*ttRanks[3] *ttDemns[3]*ttDemns[4]*ttDemns[5]* ttRanks[6]);
    assert(cudaStat3 ==cudaSuccess);
    smallargeMatrixMulti_slice_host(cublasH, h_G4, h_G56, ttRanks[3]*ttDemns[3], ttRanks[4], ttDemns[4]*ttDemns[5]*ttRanks[6], h_G456, 1, beta0);
    if(h_G56) cudaFreeHost(h_G56);h_G56=NULL;
    if(h_G4) cudaFreeHost(h_G4);h_G4=NULL;



    cudaStat1 = cudaMallocHost((void**)&h_G3, sizeof(dt)*ttRanks[2]*ttDemns[2]*ttRanks[3]);
    assert(cudaStat1 == cudaSuccess);
    for(long long i = 0; i < ttRanks[2]*ttDemns[2]*ttRanks[3]; i++){
        h_G3[i] = (dt) ((rand()*1.0) / (RAND_MAX*1.0))*2.0 - 1.0;
    }
    cudaStat3 = cudaMallocHost((void**)&h_G3456, sizeof(dt)*ttRanks[2]*ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5]*ttRanks[6]);
    assert(cudaStat3 == cudaSuccess);
    smallargeMatrixMulti_slice_host(cublasH, h_G3, h_G456, ttRanks[2]*ttDemns[2], ttRanks[3], ttDemns[3]*ttDemns[4]*ttDemns[5]*ttRanks[6], h_G3456, 1, beta0);
    if(h_G456) cudaFreeHost(h_G456);h_G456=NULL;
    if(h_G3) cudaFreeHost(h_G3);h_G3=NULL;



    cudaStat1 = cudaMallocHost((void**)&h_G2, sizeof(dt)*ttRanks[1]*ttDemns[1]*ttRanks[2]);
    assert(cudaStat1 == cudaSuccess);
    for(long long i = 0; i < ttRanks[1]*ttDemns[1]*ttRanks[2]; i++){
        h_G2[i] = (dt) ((rand()*1.0) / (RAND_MAX*1.0))*2.0 - 1.0;
    }
    cudaStat3 = cudaMallocHost((void**)&h_G23456, sizeof(dt)*ttRanks[1]*ttDemns[1]*ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5]*ttRanks[6]);
    assert(cudaStat3 == cudaSuccess);
    smallargeMatrixMulti_slice_host(cublasH, h_G2, h_G3456, ttRanks[1]*ttDemns[1], ttRanks[2], ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5]*ttRanks[6], h_G23456, 4, beta0);
    if(h_G3456) cudaFreeHost(h_G3456); h_G3456 =NULL;
    if(h_G2) cudaFreeHost(h_G2);h_G2=NULL;

    cudaStat1 = cudaMallocHost((void**)&h_G1, sizeof(dt)*ttRanks[0]*ttDemns[0]*ttRanks[1]);
    assert(cudaStat1 == cudaSuccess);
    for(long long i = 0; i < ttRanks[0]*ttDemns[0]*ttRanks[1]; i++){
        h_G1[i] = (dt) ((rand()*1.0) / (RAND_MAX*1.0))*2.0 - 1.0;
    }
    smallargeMatrixMulti_slice_host(cublasH, h_G1, h_G23456, ttRanks[0]*ttDemns[0], ttRanks[1], ttDemns[1]*ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5]*ttRanks[6], h_A, p, beta0);
    if(h_G23456) cudaFreeHost(h_G23456); h_G23456 =NULL;
    if(h_G1) cudaFreeHost(h_G1);h_G1=NULL;
    if(cublasH  ) cublasDestroy(cublasH); cublasH = NULL;
}


__global__ void matrixInvertColumn_kernel(dt *d_A, dt *d_Ainv, int m, int n){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<m*n){
        int row = i%m;
        int col = i/m;
        d_Ainv[i]=d_A[row+(m-col-1)*m];
        // d_Ainv[i] = d_A[];
    }
    __syncthreads();
}

void matrixInvertColumn(dt *d_A, dt *d_Ainv,const int &m, const int &n){
    dim3 threads(512,1,1);
    dim3 blocks((m*n+512-1)/512,1,1);
    matrixInvertColumn_kernel<<<blocks,threads>>>(d_A,d_Ainv,m,n);
}


__global__  void floattohalf(dt *AA,half *BB,long m){
    long i = blockIdx.x*blockDim.x+threadIdx.x;
    const long temp = blockDim.x*gridDim.x;
    if(i<m){
        BB[i] = __float2half(AA[i]);
        i+=temp;
    }
    __syncthreads();
}

void f2h(dt *A,half *B,long num){
    dim3 threads(1024,1,1);
    dim3 blocks((num+1024-1)/1024,1,1);   
    floattohalf<<<blocks,threads>>>(A,B,num);
}


//在TT分解中返回的是U 然后利用
void basicEig(cusolverDnHandle_t cusolverH, dt *d_A, const int &m, dt *d_W){  
    int *devInfo = NULL;
    dt *d_work = NULL;
    int  lwork = 0;
    int info_gpu = 0;

    cudaMalloc ((void**)&devInfo, sizeof(int));
    
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cusolver_status = cusolverDnSsyevd_bufferSize(
        cusolverH,
        jobz,
        uplo,
        m,
        d_A,
        m,
        d_W,
        &lwork);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(dt)*lwork);
    assert(cudaSuccess == cudaStat1);

    // printMatrix_Device(10, 10, d_A, 10, "A");
// step 4: compute spectrum
    cusolver_status = cusolverDnSsyevd(
        cusolverH,
        jobz,
        uplo,
        m,
        d_A,
        m,
        d_W,
        d_work,
        lwork,
        devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    // printMatrix_Device(10, 10, d_A, 10, "A2");
    cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    // printf("after syevd: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);

    if (devInfo) cudaFree(devInfo);
    if (d_work ) cudaFree(d_work);
    // if (d_W    ) cudaFree(d_W);
}

void largeMatrixSelfMulti_once_device(cublasHandle_t cublasH, dt *d_A,const int &m,const long long &n, dt *d_AAT){
    half *d_Ah = NULL;
    cudaStat1 = cudaMalloc((void**)&d_Ah, sizeof(half)*m*n);
    assert(cudaStat1 == cudaSuccess);
    f2h(d_A, d_Ah, m*n);
    cublas_status = cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
                           m, m, n,
                           &alpha, d_Ah, CUDA_R_16F, m,
                           d_Ah, CUDA_R_16F, m,
                           &beta0, d_AAT, CUDA_R_32F, m,
                           CUDA_R_32F, algo);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    if(d_Ah     ) cudaFree(d_Ah); d_Ah=NULL;
}

void smallargeMatrixMulti_once_device(cublasHandle_t cublasH, dt *d_A, dt *d_B, const int &ttrank, const int &m, const long long &n, dt *d_ATB){
    half *d_Ah = NULL, *d_Bh = NULL;
    cudaStat1 = cudaMalloc((void**)&d_Ah, sizeof(half)*m*ttrank);
    cudaStat2 = cudaMalloc((void**)&d_Bh, sizeof(half)*m*n);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    f2h(d_A, d_Ah, m*ttrank);
    f2h(d_B, d_Bh, m*n);

    cublas_status = cublasGemmEx(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                           ttrank, n, m,
                           &alpha, d_Ah, CUDA_R_16F, m,
                           d_Bh, CUDA_R_16F, m,
                           &beta0, d_ATB, CUDA_R_32F, ttrank,
                           CUDA_R_32F, algo);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    if(d_Ah     ) cudaFree(d_Ah); d_Ah=NULL;
    if(d_Bh     ) cudaFree(d_Bh); d_Bh=NULL;
}

void smallargeMatrixMulti_slice_device(cublasHandle_t cublasH, dt *d_A, dt *d_B, const int &ttrank, const int &m, const long long &n, dt *d_ATB, const int &p){
    half *d_Ah = NULL, *d_Bh = NULL;

    long long slice = n/p;
    cudaStat1 = cudaMalloc((void**)&d_Ah, sizeof(half)*m*ttrank);
    cudaStat2 = cudaMalloc((void**)&d_Bh, sizeof(half)*m*n);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    f2h(d_A, d_Ah, m*ttrank);
    f2h(d_B, d_Bh, m*n);
    //slice by slice
    for(int i = 0; i<p; i++){
        cublas_status = cublasGemmEx(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                           ttrank, slice, m,
                           &alpha, d_Ah, CUDA_R_16F, m,
                           d_Bh+i*m*slice, CUDA_R_16F, m,
                           &beta0, d_ATB+slice*i*ttrank, CUDA_R_32F, ttrank,
                           CUDA_R_32F, algo); 
        assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    }
    if(d_Ah     ) cudaFree(d_Ah); d_Ah=NULL;
    if(d_Bh     ) cudaFree(d_Bh); d_Bh=NULL;
}

void longMatrixSVD_Eign_once_device(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, dt *d_A, const int &m, const long long &n, const int &ttRank, dt *d_G, dt *d_A2, const int &p){
    // GPUTimer timer;
    // timer.start();
    dt *d_W=NULL, *d_AAT=NULL;
    cudaStat1 = cudaMalloc((void**)&d_AAT, sizeof(dt)*m*m);
    cudaStat2 = cudaMalloc((void**)&d_W, sizeof(dt)*m);
 
    largeMatrixSelfMulti_once_device(cublasH, d_A, m, n, d_AAT);
    basicEig(cusolverH, d_AAT, m, d_W);
    if(d_W     ) cudaFree(d_W); d_W = NULL;

    matrixInvertColumn(d_AAT, d_G, m, ttRank);
    if(d_AAT    ) cudaFree(d_AAT);d_AAT=NULL;

    if(m>24 && n>pow(m,4)){
        smallargeMatrixMulti_slice_device(cublasH, d_G, d_A, ttRank, m, n, d_A2, p);
    }
    else{
        smallargeMatrixMulti_once_device(cublasH, d_G, d_A, ttRank, m, n, d_A2);
    }
}

float ttdec_half_lu_device_6_multiGPU(dt* h_A, const int *ttRanks, const int *ttDemns, const int &calculateTimes, const int &p, bool calError){
    printf("Start mul \n");
    GPUTimer timer;
    timer.start(); 

    for(int i=0;i<calculateTimes;i++){
        // printf("***************************** %d ********************************\n", i);
///TODO 改成多GPU的逻辑 
    //1
        // GPUTimer timer0;
        // timer0.start();

        int m = ttDemns[0];
        int n = ttDemns[1]*ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5];
        int n_half = n/2;
        dt alpha = 1.0, beta = 1.0;
        // printMatrix(m,n,h_A,m,"A"); 
        dt *d_A1=NULL, *d_AAT1=NULL, *d_A21=NULL, *d_G11=NULL;
        dt *d_A0=NULL, *d_AAT0=NULL, *d_A2=NULL, *d_G10=NULL;
        dt *d_AAT=NULL, *h_G1=NULL;

        cublasHandle_t cublasH1 = NULL;
        cublasHandle_t cublasH = NULL;
        cusolverDnHandle_t cusolverH = NULL;


        cudaSetDevice(1);
        cublas_status = cublasCreate(&cublasH1);
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);
        cublas_status = cublasSetMathMode(cublasH1, mathMode);
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);

        cudaStat1 = cudaMalloc((void**)&d_A1, sizeof(dt)*m*n_half);
        cudaStat2 = cudaMalloc((void**)&d_AAT1, sizeof(dt)*m*m);
        cudaStat3 = cudaMalloc((void**)&d_A21, sizeof(dt)*ttRanks[1]*n_half);
        cudaStat4 = cudaMalloc((void**)&d_G11, sizeof(dt)*m*ttRanks[1]);
        assert(cudaStat1 == cudaSuccess);
        assert(cudaStat2 == cudaSuccess);
        // assert(cudaStat3 == cudaSuccess);

        cudaSetDevice(0);
        cublas_status = cublasCreate(&cublasH);
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);
        cublas_status = cublasSetMathMode(cublasH, mathMode);
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);

        cudaStat1 = cudaMalloc((void**)&d_A0, sizeof(dt)*m*n_half);
        cudaStat2 = cudaMalloc((void**)&d_AAT0, sizeof(dt)*m*m);
        cudaStat3 = cudaMalloc((void**)&d_AAT, sizeof(dt)*m*m);
        cudaStat4 = cudaMalloc((void**)&d_A2, sizeof(dt)*n*ttRanks[1]);
        assert(cudaStat1 == cudaSuccess);
        assert(cudaStat2 == cudaSuccess);
        assert(cudaStat3 == cudaSuccess);
        assert(cudaStat4 == cudaSuccess);
        
#pragma omp parallel num_threads(2)
        {
            int cpuid = omp_get_thread_num();
            if(cpuid == 1)
            {   
                cudaSetDevice(1);
                cudaDeviceEnablePeerAccess(0,0);
                cudaStat1 = cudaMemcpy(d_A1, h_A+m*n_half, sizeof(dt)*m*n_half, cudaMemcpyHostToDevice);
                assert(cudaStat1 == cudaSuccess);
                largeMatrixSelfMulti_once_device(cublasH1, d_A1, m, n_half, d_AAT1);
            }
            else if(cpuid == 0)
            {   
                cudaSetDevice(0);
                cudaDeviceEnablePeerAccess(1,0);
                cudaStat1 = cudaMemcpy(d_A0, h_A, sizeof(dt)*m*n_half, cudaMemcpyHostToDevice);
                assert(cudaStat1 == cudaSuccess);
                largeMatrixSelfMulti_once_device(cublasH, d_A0, m, n_half, d_AAT0);
            }
        }
        //同步数据
        cudaStat1 = cudaDeviceSynchronize();
        assert(cudaStat1 == cudaSuccess);

        cudaSetDevice(0);
        cudaStat1 = cudaMemcpyPeer(d_AAT, 0, d_AAT1, 1, sizeof(dt)*m*m);
        // cudaStat2 = cudaMemcpyPeer(d_A0+m*n_half, 0, d_A1, 1, sizeof(dt)*m*n_half);
        assert(cudaStat1 == cudaSuccess);
        // assert(cudaStat2 == cudaSuccess);

        // printMatrix_Device(m,n,d_A0,m,"A0");
        // printMatrix_Device(m,m,d_AAT1,m,"AAT1");
        // printMatrix_Device(m,m,d_AAT0,m,"AAT0");
        // printMatrix_Device(m,m,d_AAT,m,"AAT");
        cublas_status = cublasSgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, m, 
                          &alpha, d_AAT0, m,
                          &beta, d_AAT, m, d_AAT, m);
        // printMatrix_Device(m,m,d_AAT,m,"AAT");
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);
        
        cusolver_status = cusolverDnCreate(&cusolverH);
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

        cudaStat1 = cudaHostAlloc((void**)&h_G1,sizeof(dt)*m*ttRanks[1],0);
        cudaStat2 = cudaMalloc((void**)&d_G10, sizeof(dt)*m*ttRanks[1]);
        assert(cudaStat1 == cudaSuccess);
        assert(cudaStat2 == cudaSuccess);
        // assert(cudaStat3 == cudaSuccess);

        dt *d_W=NULL;
        cudaStat2 = cudaMalloc((void**)&d_W, sizeof(dt)*m);
        
        basicEig(cusolverH, d_AAT, m, d_W);
        // printf("basicEig Finish\n");
        if(d_W     ) cudaFree(d_W); d_W = NULL;

        // printf("matrixInvertColumn\n");
        // GPUTimer timer2;
        // timer2.start();
        matrixInvertColumn(d_AAT, d_G10, m, ttRanks[1]);

        // printMatrix_Device(m,ttRanks[1],d_G10,m,"G10");

        if(d_AAT1      ) cudaFree(d_AAT1); d_AAT1 = NULL;
        if(d_AAT0     ) cudaFree(d_AAT0); d_AAT0 = NULL;
        if(d_AAT     ) cudaFree(d_AAT); d_AAT = NULL;

        cudaStat5 = cudaMemcpyAsync(h_G1, d_G10, sizeof(dt)*m*ttRanks[1], cudaMemcpyDeviceToHost,0);
        assert(cudaStat5 == cudaSuccess);

        // 大于24需要分片 
        // GPUTimer timer3;
        // timer3.start();

        cudaStat2 = cudaMemcpyPeer(d_G11, 1, d_G10, 0, sizeof(dt)*m*ttRanks[1]);
        assert(cudaStat2 == cudaSuccess);
        // printMatrix_Device(m,ttRanks[1],d_G11,m,"G11");


#pragma omp parallel num_threads(2)
        {
            int cpuid = omp_get_thread_num();
            if(cpuid == 1)
            {      
                cudaSetDevice(1);

                if(m>24 && n>pow(m,4)){
                    // printf("smallargeMatrixMulti_slice_device\n");
                    smallargeMatrixMulti_slice_device(cublasH1, d_G11, d_A1, ttRanks[1], m, n_half, d_A21, p);
                    // printMatrix_Device(10, 10, d_A2, 10, "A2");
                }
                else{
                    // printf("smallargeMatrixMulti_once_device\n");
                    smallargeMatrixMulti_once_device(cublasH1, d_G11, d_A1, ttRanks[1], m, n_half, d_A21);
                }            
            }
            else if(cpuid == 0)
            {   
                cudaSetDevice(0);

                if(m>24 && n>pow(m,4)){
                    // printf("smallargeMatrixMulti_slice_device\n");
                    smallargeMatrixMulti_slice_device(cublasH, d_G10, d_A0, ttRanks[1], m, n_half, d_A2, p);
                        // printMatrix_Device(10, 10, d_A2, 10, "A2");
                }
                else{
                        // printf("smallargeMatrixMulti_once_device\n");
                    smallargeMatrixMulti_once_device(cublasH, d_G10, d_A0, ttRanks[1], m, n_half, d_A2);
                }
            }
        }
        cudaStat1 = cudaDeviceSynchronize();
        assert(cudaStat1 == cudaSuccess);
        cudaSetDevice(0);
        cudaStat1 = cudaMemcpyPeer(d_A2+n_half*ttRanks[1], 0, d_A21, 1, sizeof(dt)*n_half*ttRanks[1]);
        assert(cudaStat1 == cudaSuccess);
        // printMatrix_Device(ttRanks[1],n,d_A2,ttRanks[1],"A2");

        if(d_A0      ) cudaFree(d_A0); d_A0 = NULL;
        if(d_A1      ) cudaFree(d_A1); d_A1 = NULL;
        if(d_A21     ) cudaFree(d_A21); d_A21 = NULL;
        if(d_G10     ) cudaFree(d_G10); d_G10 = NULL;
        if(d_G11     ) cudaFree(d_G11); d_G11 = NULL;
        // printf(" multi GPU  first time： %f\n\n", timer0.seconds());

    //1->2
        // GPUTimer timer1;
        // timer1.start();
        dt *d_G2=NULL, *d_A3=NULL, *h_G2=NULL;
        m = ttRanks[1] * ttDemns[1]; //4
        n = n/ttDemns[1]; //16
        cudaStat1 = cudaMalloc((void**)&d_G2, sizeof(dt)*m*ttRanks[2]);
        cudaStat2 = cudaHostAlloc((void**)&h_G2,sizeof(dt)*m*ttRanks[2],0);
        cudaStat2 = cudaMalloc((void**)&d_A3, sizeof(dt)*n*ttRanks[2]);
        assert(cudaStat1 == cudaSuccess);
        assert(cudaStat2 == cudaSuccess);
        // printMatrix_Device(10, 10, d_A2, 10, "A22");
        // printMatrix_Device(m, n, d_A2, m, "A22");
        longMatrixSVD_Eign_once_device(cublasH, cusolverH, d_A2, m, n, ttRanks[2] , d_G2, d_A3, p);
        // printMatrix_Device(10, 10, d_G2, 10, "G2");
        // printMatrix_Device(10, 10, d_A3, 10, "A3");
        // printMatrix_Device(m, ttRanks[2], d_G2, m, "G2");
        // printMatrix_Device(ttRanks[2], n, d_A3, ttRanks[2], "A3");
        if(d_A2     ) cudaFree(d_A2); d_A2 = NULL;
        cudaStat5 = cudaMemcpyAsync(h_G2, d_G2, sizeof(dt)*m*ttRanks[2], cudaMemcpyDeviceToHost,0);
        // cudaStat6 = cudaDeviceSynchronize();
        assert(cudaStat5 == cudaSuccess);
        // assert(cudaStat6 == cudaSuccess);
        if(d_G2     ) cudaFree(d_G2); d_G2 = NULL;
        // printf(" second time： %f\n\n", timer1.seconds());

    //1->2->3
        // GPUTimer timer2;
        // timer2.start();
        dt *d_G3=NULL, *d_A4=NULL, *h_G3=NULL;
        m = ttRanks[2] * ttDemns[2];
        n = n/ttDemns[2];
        cudaStat1 = cudaMalloc((void**)&d_G3, sizeof(dt)*m*ttRanks[3]);
        cudaStat2 = cudaHostAlloc((void**)&h_G3,sizeof(dt)*m*ttRanks[3],0);
        cudaStat2 = cudaMalloc((void**)&d_A4, sizeof(dt)*ttRanks[3]*n);
        assert(cudaStat1 == cudaSuccess);
        assert(cudaStat2 == cudaSuccess);
        // printMatrix_Device(10, 10, d_A3, 10, "A32");
        // printMatrix_Device(m, n, d_A3, m, "A32");
        longMatrixSVD_Eign_once_device(cublasH, cusolverH, d_A3, m, n, ttRanks[3] , d_G3, d_A4, p);
        // printMatrix_Device(10, 10, d_G3, 10, "G3");
        // printMatrix_Device(10, 10, d_A4, 10, "A4");
        // printMatrix_Device(m, ttRanks[3], d_G3, m, "G3");
        // printMatrix_Device(ttRanks[3], n, d_A4, ttRanks[3], "A4");
        if(d_A3     ) cudaFree(d_A3); d_A3 = NULL;
        cudaStat5 = cudaMemcpyAsync(h_G3, d_G3, sizeof(dt)*m*ttRanks[3], cudaMemcpyDeviceToHost,0);
        // cudaStat6 = cudaDeviceSynchronize();
        assert(cudaStat5 == cudaSuccess);
        // assert(cudaStat6 == cudaSuccess);
        if(d_G3     ) cudaFree(d_G3); d_G3 = NULL;
        // printf(" third time： %f\n\n", timer2.seconds());

    //1->2->3->4
        // GPUTimer timer3;
        // timer3.start();
        dt *d_G4=NULL, *d_A5=NULL, *h_G4=NULL;
        m = ttRanks[3] * ttDemns[3];
        n = n/ttDemns[3];
        cudaStat1 = cudaMalloc((void**)&d_G4, sizeof(dt)*m*ttRanks[4]);
        cudaStat2 = cudaHostAlloc((void**)&h_G4,sizeof(dt)*m*ttRanks[4],0);
        cudaStat2 = cudaMalloc((void**)&d_A5, sizeof(dt)*ttRanks[4]*n);
        assert(cudaStat1 == cudaSuccess);
        assert(cudaStat2 == cudaSuccess);
        // printMatrix_Device(10, 10, d_A4, 10, "A42");
        // printMatrix_Device(m, n, d_A4, m, "A42");
        longMatrixSVD_Eign_once_device(cublasH, cusolverH, d_A4, m, n, ttRanks[4] , d_G4, d_A5, p);
        // printMatrix_Device(10, 10, d_G4, 10, "G4");
        // printMatrix_Device(10, 10, d_A5, 10, "A5");
        // printMatrix_Device(m, ttRanks[4], d_G4, m, "G4");
        // printMatrix_Device(ttRanks[4], n, d_A5, ttRanks[4], "A5");
        if(d_A4     ) cudaFree(d_A4); d_A4 = NULL;
        cudaStat5 = cudaMemcpyAsync(h_G4, d_G4, sizeof(dt)*m*ttRanks[4], cudaMemcpyDeviceToHost,0);
        // cudaStat6 = cudaDeviceSynchronize();
        assert(cudaStat5 == cudaSuccess);
        // assert(cudaStat6 == cudaSuccess);
        if(d_G4     ) cudaFree(d_G4); d_G4 = NULL;
        // printf(" forth time： %f\n\n", timer3.seconds());

    //1->2->3->4->5
        // GPUTimer timer4;
        // timer4.start();
        dt *d_G5=NULL, *d_G6=NULL, *h_G5=NULL, *h_G6=NULL;
        m = ttRanks[4] * ttDemns[4];
        n = n/ttDemns[4];
        cudaStat1 = cudaMalloc((void**)&d_G5, sizeof(dt)*m*ttRanks[5]);
        cudaStat2 = cudaHostAlloc((void**)&h_G5,sizeof(dt)*m*ttRanks[5],0);
        cudaStat2 = cudaMalloc((void**)&d_G6, sizeof(dt)*ttRanks[5]*n);
        cudaStat2 = cudaHostAlloc((void**)&h_G6,sizeof(dt)*n*ttRanks[5],0);
        assert(cudaStat1 == cudaSuccess);
        assert(cudaStat2 == cudaSuccess);
        // printMatrix_Device(10, 10, d_A5, 10, "A52");
        // printMatrix_Device(m, n, d_A5, m, "A52");
        longMatrixSVD_Eign_once_device(cublasH, cusolverH, d_A5, m, n, ttRanks[5] , d_G5, d_G6, p);
        // printMatrix_Device(10, 10, d_G5, 10, "G5");
        // printMatrix_Device(10, 10, d_G6, 10, "G6");
        // printMatrix_Device(m, ttRanks[5], d_G5, m, "G5");
        // printMatrix_Device(ttRanks[5], n, d_G6, ttRanks[5], "A6");
        if(d_A5     ) cudaFree(d_A5); d_A5 = NULL;
        cudaStat5 = cudaMemcpyAsync(h_G5, d_G5, sizeof(dt)*m*ttRanks[5], cudaMemcpyDeviceToHost,0);
        cudaStat6 = cudaMemcpyAsync(h_G6, d_G6, sizeof(dt)*n*ttRanks[5], cudaMemcpyDeviceToHost,0);
        // cudaStat6 = cudaDeviceSynchronize();
        assert(cudaStat5 == cudaSuccess);
        assert(cudaStat6 == cudaSuccess);
        if(d_G5     ) cudaFree(d_G5); d_G5 = NULL;
        if(d_G6     ) cudaFree(d_G6); d_G6 = NULL;
        // printf(" fifth time： %f\n\n", timer4.seconds());

        if(calError){
            // printMatrix(10,10,h_A,10,"hA");
            double MSE = calMSE(cublasH, h_A, h_G1, h_G2, h_G3, h_G4, h_G5, h_G6, ttRanks, ttDemns, p);
            printf("\n*********************** MSE: %f ******************\n", MSE);
        }

        if(h_G1     ) cudaFreeHost(h_G1);h_G1 = NULL;
        if(h_G2     ) cudaFreeHost(h_G2);h_G2 = NULL;
        if(h_G3     ) cudaFreeHost(h_G3);h_G3 = NULL;
        if(h_G4     ) cudaFreeHost(h_G4);h_G4 = NULL;
        if(h_G5     ) cudaFreeHost(h_G5);h_G5 = NULL;
        if(h_G6     ) cudaFreeHost(h_G6);h_G6 = NULL;
        if(cublasH  ) cublasDestroy(cublasH);
        if(cusolverH) cusolverDnDestroy(cusolverH); 
    }
    float time = timer.seconds()/calculateTimes;
    return time;
}

// , const long long &over, const long long &m,const long long &n, 
// 实现一个超大规模矩阵A的 A*AT，存储到GPU上
void largeMatrixSelfMulti_slice_host(cublasHandle_t cublasH, dt *h_A, const long long &over, const long long &m, const long long &n, dt *d_AAT, const int &p, const int & streamNumber){
    long long slice = 0, le = 0;

    if(n%p==0){
        // printf("n chuyi p is 0, and threads:%d \n", omp_get_thread_num());
        slice = n/p;
    }else{
        slice = n/p;
        le = n%p;
    }

    float *d_tempA =NULL; 
    half *d_Ah = NULL;
    cudaStat1 = cudaMalloc((void**)&d_tempA, sizeof(dt)*m*slice);
    cudaStat2 = cudaMalloc((void**)&d_Ah, sizeof(half)*m*slice);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    // printf("slice is %lld\n", slice);

    for(long long i=0;i<p;i++){
        ///有问题需要修改
        long long overTemp = over+m*slice*i;

        cudaStat3 = cudaMemcpy(d_tempA, h_A+overTemp, sizeof(dt)*m*slice, cudaMemcpyHostToDevice);

        if (cudaStat3 != cudaSuccess){
            printf("over: %lld, m: %lld, slice: %lld, i: %lld", over, m, slice, i);

            printf("530 error %d, overTemp: %lld , slice: %lld , i : %lld, m : %lld , n : %lld , p : %d, deviceid : %d\n", cudaStat3, overTemp, slice, i, m, n, p, streamNumber);
        }
        assert(cudaStat3 == cudaSuccess);
        // assert(cudaStat4 == cudaSuccess);
        //判断是否降维
        f2h(d_tempA, d_Ah, m*slice);
        cublas_status = cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
                           m, m, slice,
                           &alpha, d_Ah, CUDA_R_16F, m,
                           d_Ah, CUDA_R_16F, m,
                           &beta0, d_AAT, CUDA_R_32F, m,
                           CUDA_R_32F, algo);
        assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    }
    if(d_Ah     ) cudaFree(d_Ah); d_Ah=NULL;
    if(d_tempA  ) cudaFree(d_tempA); d_tempA=NULL;
}


void smallargeMatrixMulti_slice_host_Tans(cublasHandle_t cublasH, dt *d_A, dt *h_B, const int &ttrank, const long long &m,  const long long &n, dt *h_ATB, const int &p){
    long long slice = 0;
    if(n%p==0)
        slice = n/p;
    else
        assert("P is not slice");
    // printf("ttrank: %d, n: %d",m,ttrank,n);
    float *d_tempB = NULL, *d_tempATB = NULL; 
    half *d_Ah = NULL, *d_Bh = NULL;
    cudaStat1 = cudaMalloc((void**)&d_tempB, sizeof(dt)*m*slice);
    cudaStat2 = cudaMalloc((void**)&d_tempATB, sizeof(dt)*ttrank*slice);
    cudaStat3 = cudaMalloc((void**)&d_Ah, sizeof(half)*m*ttrank);
    cudaStat4 = cudaMalloc((void**)&d_Bh, sizeof(half)*m*slice);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);
    assert(cudaStat4 == cudaSuccess);

    // printMatrix_Device(m, ttrank, d_A, m, "tempA");
    f2h(d_A, d_Ah, m*ttrank);
    // printf("%d\n", slice);
    for(long long i=0;i<p;i++){
        cudaStat3 = cudaMemcpyAsync(d_tempB, h_B+m*slice*i, sizeof(dt)*m*slice, cudaMemcpyHostToDevice,0);
        cudaStat4 = cudaDeviceSynchronize();
        assert(cudaStat3 == cudaSuccess);
        assert(cudaStat4 == cudaSuccess);
        f2h(d_tempB, d_Bh, m*slice);
        cublas_status = cublasGemmEx(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                           ttrank, slice, m,
                           &alpha, d_Ah, CUDA_R_16F, m,
                           d_Bh, CUDA_R_16F, m,
                           &beta0, d_tempATB, CUDA_R_32F, ttrank,
                           CUDA_R_32F, algo);
        assert(cublas_status == CUBLAS_STATUS_SUCCESS);
        // printMatrix_Device(ttrank, slice, d_tempATB, ttrank, "tempATB");

        cudaStat3 = cudaMemcpyAsync(h_ATB+ttrank*slice*i, d_tempATB, sizeof(dt)*ttrank*slice, cudaMemcpyDeviceToHost, 0);
        cudaStat4 = cudaDeviceSynchronize();
        assert(cudaStat3 == cudaSuccess);
        assert(cudaStat4 == cudaSuccess);
    }
    if(d_Ah     ) cudaFree(d_Ah); d_Ah=NULL;
    if(d_Bh     ) cudaFree(d_Bh); d_Bh=NULL;
    if(d_tempB  ) cudaFree(d_tempB); d_tempB=NULL;
    if(d_tempATB  ) cudaFree(d_tempATB); d_tempATB=NULL;
}


void multi_gpu_single_order_decomposition(){
    
}


float ttdec_half_lu_host_6_multiGPU(dt* h_A, const int *ttRanks, const int *ttDemns, const int &calculateTimes, const int &p, bool calError){
    printf("Start mul \n");
    GPUTimer timer;
    timer.start(); 
    int device[8] = {0,1,2,3,4,5,6,7};
    for(int i=0;i<calculateTimes;i++){
        cudaSetDevice(device[0]);
        printf("***************************** %d ********************************\n", i);
    //1
        int m = ttDemns[0];
        int n = ttDemns[1]*ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5];
        int n_half = n/8;
        printf("n_half: %d", n_half);
        dt alpha = 1.0, beta = 1.0;
        // printMatrix(m,n,h_A,m,"A"); 
        dt *d_AAT0=NULL, *d_A2=NULL, *d_G10=NULL;
        dt *d_AAT1=NULL, *d_A21=NULL, *d_G11=NULL;
        dt *d_AAT2=NULL, *d_A22=NULL, *d_G12=NULL;
        dt *d_AAT3=NULL, *d_A23=NULL, *d_G13=NULL;
        dt *d_AAT4=NULL, *d_A24=NULL, *d_G14=NULL;
        dt *d_AAT5=NULL, *d_A25=NULL, *d_G15=NULL;
        dt *d_AAT6=NULL, *d_A26=NULL, *d_G16=NULL;
        dt *d_AAT7=NULL, *d_A27=NULL, *d_G17=NULL;
        dt *d_AAT=NULL, *h_G1=NULL;

        cublasHandle_t cublasH = NULL;
        cublasHandle_t cublasH1 = NULL;
        cublasHandle_t cublasH2 = NULL;
        cublasHandle_t cublasH3 = NULL;
        cublasHandle_t cublasH4 = NULL;
        cublasHandle_t cublasH5 = NULL;
        cublasHandle_t cublasH6 = NULL;
        cublasHandle_t cublasH7 = NULL;
        cusolverDnHandle_t cusolverH = NULL;

#pragma omp parallel num_threads(8)
        {
            int cpuid = omp_get_thread_num();
            if(cpuid == 7)
            {   
                cudaSetDevice(device[7]);
                cublas_status = cublasCreate(&cublasH7);
                assert(CUBLAS_STATUS_SUCCESS == cublas_status);
                cublas_status = cublasSetMathMode(cublasH7, mathMode);
                assert(CUBLAS_STATUS_SUCCESS == cublas_status);

                cudaStat1 = cudaMalloc((void**)&d_AAT7, sizeof(dt)*m*m);
                cudaStat2 = cudaMalloc((void**)&d_A27, sizeof(dt)*ttRanks[1]*n_half);
                cudaStat3 = cudaMalloc((void**)&d_G17, sizeof(dt)*m*ttRanks[1]);
                assert(cudaStat1 == cudaSuccess);
                assert(cudaStat2 == cudaSuccess);
                assert(cudaStat3 == cudaSuccess);
                cudaDeviceEnablePeerAccess(0,0);
                long long over1 = ((long long)m)*n_half*(long long)7;
                largeMatrixSelfMulti_slice_host(cublasH7, h_A, over1, (long long)m, n_half, d_AAT7, p, device[7]);
                printf("thread 7 finish\n");
            }
            else if(cpuid == 6)
            {   
                cudaSetDevice(device[6]);
                cublas_status = cublasCreate(&cublasH6);
                assert(CUBLAS_STATUS_SUCCESS == cublas_status);
                cublas_status = cublasSetMathMode(cublasH6, mathMode);
                assert(CUBLAS_STATUS_SUCCESS == cublas_status);

                cudaStat2 = cudaMalloc((void**)&d_AAT6, sizeof(dt)*m*m);
                cudaStat3 = cudaMalloc((void**)&d_A26, sizeof(dt)*ttRanks[1]*n_half);
                cudaStat4 = cudaMalloc((void**)&d_G16, sizeof(dt)*m*ttRanks[1]);
                assert(cudaStat1 == cudaSuccess);
                assert(cudaStat2 == cudaSuccess);

                cudaDeviceEnablePeerAccess(0,0);
                long long over1 = ((long long)m)*n_half*(long long)6;
                largeMatrixSelfMulti_slice_host(cublasH6, h_A, over1, (long long)m, n_half, d_AAT6, p, device[6]);
                printf("thread 6 finish\n");
            }
            else if(cpuid == 5)
            {   
                cudaSetDevice(device[5]);
                cublas_status = cublasCreate(&cublasH5);
                assert(CUBLAS_STATUS_SUCCESS == cublas_status);
                cublas_status = cublasSetMathMode(cublasH5, mathMode);
                assert(CUBLAS_STATUS_SUCCESS == cublas_status);

                cudaStat1 = cudaMalloc((void**)&d_AAT5, sizeof(dt)*m*m);
                cudaStat2 = cudaMalloc((void**)&d_A25, sizeof(dt)*ttRanks[1]*n_half);
                cudaStat3 = cudaMalloc((void**)&d_G15, sizeof(dt)*m*ttRanks[1]);
                assert(cudaStat1 == cudaSuccess);
                assert(cudaStat2 == cudaSuccess);
                assert(cudaStat3 == cudaSuccess);

                cudaDeviceEnablePeerAccess(0,0);
                long long over1 = ((long long)m)*n_half*(long long)5;
                largeMatrixSelfMulti_slice_host(cublasH5, h_A, over1, (long long)m, n_half, d_AAT5, p, device[5]);
                printf("thread 5 finish\n");
            }
            else if(cpuid == 4)
            {   
                cudaSetDevice(device[4]);
                cublas_status = cublasCreate(&cublasH4);
                assert(CUBLAS_STATUS_SUCCESS == cublas_status);
                cublas_status = cublasSetMathMode(cublasH4, mathMode);
                assert(CUBLAS_STATUS_SUCCESS == cublas_status);

                cudaStat1 = cudaMalloc((void**)&d_AAT4, sizeof(dt)*m*m);
                cudaStat2 = cudaMalloc((void**)&d_A24, sizeof(dt)*ttRanks[1]*n_half);
                cudaStat3 = cudaMalloc((void**)&d_G14, sizeof(dt)*m*ttRanks[1]);
                assert(cudaStat1 == cudaSuccess);
                assert(cudaStat2 == cudaSuccess);
                assert(cudaStat3 == cudaSuccess);

                cudaDeviceEnablePeerAccess(0,0);
                long long over1 = ((long long)m)*n_half*(long long)4;
                largeMatrixSelfMulti_slice_host(cublasH4, h_A, over1, (long long)m, n_half, d_AAT4, p, device[4]);
                printf("thread 4 finish\n");
            }
            else if(cpuid == 3)
            {   
                cudaSetDevice(device[3]);
                cublas_status = cublasCreate(&cublasH3);
                assert(CUBLAS_STATUS_SUCCESS == cublas_status);
                cublas_status = cublasSetMathMode(cublasH3, mathMode);
                assert(CUBLAS_STATUS_SUCCESS == cublas_status);

                cudaStat1 = cudaMalloc((void**)&d_AAT3, sizeof(dt)*m*m);
                cudaStat2 = cudaMalloc((void**)&d_A23, sizeof(dt)*ttRanks[1]*n_half);
                cudaStat3 = cudaMalloc((void**)&d_G13, sizeof(dt)*m*ttRanks[1]);
                assert(cudaStat1 == cudaSuccess);
                assert(cudaStat2 == cudaSuccess);
                assert(cudaStat3 == cudaSuccess);

                cudaDeviceEnablePeerAccess(0,0);
                long long over1 = ((long long)m)*n_half*(long long)3;
                largeMatrixSelfMulti_slice_host(cublasH3, h_A, over1, (long long)m, n_half, d_AAT3, p, device[3]);
                printf("thread 3 finish\n");
            }
            else if(cpuid == 2)
            {   
                cudaSetDevice(device[2]);
                cublas_status = cublasCreate(&cublasH2);
                assert(CUBLAS_STATUS_SUCCESS == cublas_status);
                cublas_status = cublasSetMathMode(cublasH2, mathMode);
                assert(CUBLAS_STATUS_SUCCESS == cublas_status);

                cudaStat1 = cudaMalloc((void**)&d_AAT2, sizeof(dt)*m*m);
                cudaStat2 = cudaMalloc((void**)&d_A22, sizeof(dt)*ttRanks[1]*n_half);
                cudaStat3 = cudaMalloc((void**)&d_G12, sizeof(dt)*m*ttRanks[1]);
                assert(cudaStat1 == cudaSuccess);
                assert(cudaStat2 == cudaSuccess);
                assert(cudaStat3 == cudaSuccess);

                cudaDeviceEnablePeerAccess(0,0);
                long long over1 = ((long long)m)*n_half*(long long)2;
                largeMatrixSelfMulti_slice_host(cublasH2, h_A, over1, (long long)m, n_half, d_AAT2, p, device[2]);
                printf("thread 2 finish\n");
            }
            else if(cpuid == 1)
            {   
                cudaSetDevice(device[1]);
                cublas_status = cublasCreate(&cublasH1);
                assert(CUBLAS_STATUS_SUCCESS == cublas_status);
                cublas_status = cublasSetMathMode(cublasH1, mathMode);
                assert(CUBLAS_STATUS_SUCCESS == cublas_status);

                cudaStat1 = cudaMalloc((void**)&d_AAT1, sizeof(dt)*m*m);
                cudaStat2 = cudaMalloc((void**)&d_A21, sizeof(dt)*ttRanks[1]*n_half);
                cudaStat2 = cudaMalloc((void**)&d_G11, sizeof(dt)*m*ttRanks[1]);
                assert(cudaStat1 == cudaSuccess);
                assert(cudaStat2 == cudaSuccess);
                assert(cudaStat3 == cudaSuccess);

                cudaDeviceEnablePeerAccess(0,0);
                long long over1 = ((long long)m)*n_half*(long long)1;
                largeMatrixSelfMulti_slice_host(cublasH1, h_A, over1, (long long)m, n_half, d_AAT1, p, device[1]);
                printf("thread 1 finish\n");
            }
            else if(cpuid == 0)
            {   
                cudaSetDevice(device[0]);
                cublas_status = cublasCreate(&cublasH);
                assert(CUBLAS_STATUS_SUCCESS == cublas_status);
                cublas_status = cublasSetMathMode(cublasH, mathMode);
                assert(CUBLAS_STATUS_SUCCESS == cublas_status);

                cudaStat1 = cudaMalloc((void**)&d_AAT0, sizeof(dt)*m*m);
                cudaStat2 = cudaMalloc((void**)&d_AAT, sizeof(dt)*m*m);
                cudaStat3 = cudaMalloc((void**)&d_A2, sizeof(dt)*n*ttRanks[1]);
                assert(cudaStat1 == cudaSuccess);
                assert(cudaStat2 == cudaSuccess);
                assert(cudaStat3 == cudaSuccess);

                cudaDeviceEnablePeerAccess(device[3],device[0]);
                cudaDeviceEnablePeerAccess(device[2],device[0]);
                cudaDeviceEnablePeerAccess(device[1],device[0]);
                largeMatrixSelfMulti_slice_host(cublasH, h_A, 0, (long long)m, n_half, d_AAT0, p, device[0]);
                printf("thread 0 finish\n");
            }
        }
        //同步数据
        cudaStat1 = cudaDeviceSynchronize();
        assert(cudaStat1 == cudaSuccess);

        cudaSetDevice(device[0]);
        cudaStat1 = cudaMemcpyPeer(d_AAT, device[0], d_AAT1, device[1], sizeof(dt)*m*m);
        assert(cudaStat1 == cudaSuccess);

        cublas_status = cublasSgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, m, 
                          &alpha, d_AAT0, m,
                          &beta, d_AAT, m, d_AAT, m);
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);

        cudaStat1 = cudaMemcpyPeer(d_AAT0, device[0], d_AAT2, device[2], sizeof(dt)*m*m);
        assert(cudaStat1 == cudaSuccess);

        cublas_status = cublasSgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, m, 
                          &alpha, d_AAT0, m,
                          &beta, d_AAT, m, d_AAT, m);
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);

        cudaStat1 = cudaMemcpyPeer(d_AAT0, device[0], d_AAT3, device[3], sizeof(dt)*m*m);
        assert(cudaStat1 == cudaSuccess);

        cublas_status = cublasSgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, m, 
                          &alpha, d_AAT0, m,
                          &beta, d_AAT, m, d_AAT, m);
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);

        cudaStat1 = cudaMemcpyPeer(d_AAT0, device[0], d_AAT4, device[4], sizeof(dt)*m*m);
        assert(cudaStat1 == cudaSuccess);

        cublas_status = cublasSgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, m, 
                          &alpha, d_AAT0, m,
                          &beta, d_AAT, m, d_AAT, m);
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);

        cudaStat1 = cudaMemcpyPeer(d_AAT0, device[0], d_AAT5, device[5], sizeof(dt)*m*m);
        assert(cudaStat1 == cudaSuccess);

        cublas_status = cublasSgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, m, 
                          &alpha, d_AAT0, m,
                          &beta, d_AAT, m, d_AAT, m);
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);

        cudaStat1 = cudaMemcpyPeer(d_AAT0, device[0], d_AAT6, device[6], sizeof(dt)*m*m);
        assert(cudaStat1 == cudaSuccess);

        cublas_status = cublasSgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, m, 
                          &alpha, d_AAT0, m,
                          &beta, d_AAT, m, d_AAT, m);
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);

        cudaStat1 = cudaMemcpyPeer(d_AAT0, device[0], d_AAT7, device[7], sizeof(dt)*m*m);
        assert(cudaStat1 == cudaSuccess);

        cublas_status = cublasSgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, m, 
                          &alpha, d_AAT0, m,
                          &beta, d_AAT, m, d_AAT, m);
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);


        
        cusolver_status = cusolverDnCreate(&cusolverH);
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

        cudaStat1 = cudaHostAlloc((void**)&h_G1,sizeof(dt)*m*ttRanks[1],0);
        cudaStat2 = cudaMalloc((void**)&d_G10, sizeof(dt)*m*ttRanks[1]);
        assert(cudaStat1 == cudaSuccess);
        assert(cudaStat2 == cudaSuccess);

        dt *d_W=NULL;
        cudaStat2 = cudaMalloc((void**)&d_W, sizeof(dt)*m);
        
        basicEig(cusolverH, d_AAT, m, d_W);
        if(d_W     ) cudaFree(d_W); d_W = NULL;

        matrixInvertColumn(d_AAT, d_G10, m, ttRanks[1]);


        if(d_AAT1      ) cudaFree(d_AAT1); d_AAT1 = NULL;
        if(d_AAT0     ) cudaFree(d_AAT0); d_AAT0 = NULL;
        if(d_AAT     ) cudaFree(d_AAT); d_AAT = NULL;

        cudaStat5 = cudaMemcpyAsync(h_G1, d_G10, sizeof(dt)*m*ttRanks[1], cudaMemcpyDeviceToHost,0);
        assert(cudaStat5 == cudaSuccess);


        #pragma omp parallel num_threads(8)
        {
            int cpuid = omp_get_thread_num();
            if(cpuid == 7)
            {      
                cudaSetDevice(device[7]);

                cudaStat2 = cudaMemcpyPeer(d_G17, device[7], d_G10, device[0], sizeof(dt)*m*ttRanks[1]);
                assert(cudaStat2 == cudaSuccess);

                smallargeMatrixMulti_slice_host_Tans(cublasH7, d_G17, h_A+((long long)m)*n_half*(long long)7, ttRanks[1], m, n_half, d_A27, p);
                printf("thread 6 finish0\n");
                if(d_G17     ) cudaFree(d_G17); d_G17 = NULL;

                cudaStat1 = cudaMemcpyPeer(d_A2+n_half*ttRanks[1]*(long long)7, device[0], d_A27, device[7], sizeof(dt)*n_half*ttRanks[1]);
                assert(cudaStat1 == cudaSuccess);
                if(d_A27     ) cudaFree(d_A27); d_A27 = NULL;
            }
            else if(cpuid == 6)
            {      
                cudaSetDevice(device[6]);

                cudaStat2 = cudaMemcpyPeer(d_G16, device[6], d_G10, device[0], sizeof(dt)*m*ttRanks[1]);
                assert(cudaStat2 == cudaSuccess);

                smallargeMatrixMulti_slice_host_Tans(cublasH6, d_G16, h_A+((long long)m)*n_half*(long long)6, ttRanks[1], m, n_half, d_A26, p);
                printf("thread 6 finish0\n");
                if(d_G16     ) cudaFree(d_G16); d_G16 = NULL;

                cudaStat1 = cudaMemcpyPeer(d_A2+n_half*ttRanks[1]*(long long)6, device[0], d_A26, device[6], sizeof(dt)*n_half*ttRanks[1]);
                assert(cudaStat1 == cudaSuccess);
                if(d_A26     ) cudaFree(d_A26); d_A26 = NULL;
            }
            else if(cpuid == 5)
            {      
                cudaSetDevice(device[5]);

                cudaStat2 = cudaMemcpyPeer(d_G15, device[5], d_G10, device[0], sizeof(dt)*m*ttRanks[1]);
                assert(cudaStat2 == cudaSuccess);

                smallargeMatrixMulti_slice_host_Tans(cublasH5, d_G15, h_A+((long long)m)*n_half*(long long)5, ttRanks[1], m, n_half, d_A25, p);
                printf("thread 3 finish0\n");
                if(d_G15     ) cudaFree(d_G15); d_G15 = NULL;

                cudaStat1 = cudaMemcpyPeer(d_A2+n_half*ttRanks[1]*(long long)5, device[0], d_A25, device[5], sizeof(dt)*n_half*ttRanks[1]);
                assert(cudaStat1 == cudaSuccess);
                if(d_A25     ) cudaFree(d_A25); d_A25 = NULL;
            }
            else if(cpuid == 4)
            {      
                cudaSetDevice(device[4]);

                cudaStat2 = cudaMemcpyPeer(d_G14, device[4], d_G10, device[0], sizeof(dt)*m*ttRanks[1]);
                assert(cudaStat2 == cudaSuccess);

                smallargeMatrixMulti_slice_host_Tans(cublasH4, d_G14, h_A+((long long)m)*n_half*(long long)4, ttRanks[1], m, n_half, d_A24, p);
                printf("thread 3 finish0\n");
                if(d_G14     ) cudaFree(d_G14); d_G14 = NULL;

                cudaStat1 = cudaMemcpyPeer(d_A2+n_half*ttRanks[1]*(long long)4, device[0], d_A24, device[4], sizeof(dt)*n_half*ttRanks[1]);
                assert(cudaStat1 == cudaSuccess);
                if(d_A24     ) cudaFree(d_A24); d_A24 = NULL;
            }
            else if(cpuid == 3)
            {      
                cudaSetDevice(device[3]);

                cudaStat2 = cudaMemcpyPeer(d_G13, device[3], d_G10, device[0], sizeof(dt)*m*ttRanks[1]);
                assert(cudaStat2 == cudaSuccess);

                smallargeMatrixMulti_slice_host_Tans(cublasH3, d_G13, h_A+((long long)m)*n_half*(long long)3, ttRanks[1], m, n_half, d_A23, p);
                printf("thread 3 finish0\n");
                if(d_G13     ) cudaFree(d_G13); d_G13 = NULL;

                cudaStat1 = cudaMemcpyPeer(d_A2+n_half*ttRanks[1]*(long long)3, device[0], d_A23, device[3], sizeof(dt)*n_half*ttRanks[1]);
                assert(cudaStat1 == cudaSuccess);
                if(d_A23     ) cudaFree(d_A23); d_A23 = NULL;
            }
            else if(cpuid == 2)
            {      
                cudaSetDevice(device[2]);

                cudaStat2 = cudaMemcpyPeer(d_G12, device[2], d_G10, device[0], sizeof(dt)*m*ttRanks[1]);
                assert(cudaStat2 == cudaSuccess);

                smallargeMatrixMulti_slice_host_Tans(cublasH2, d_G12, h_A+((long long)m)*n_half*(long long)2, ttRanks[1], m, n_half, d_A22, p);
                printf("thread 2 finish0\n");
                if(d_G12     ) cudaFree(d_G12); d_G12 = NULL;

                cudaStat1 = cudaMemcpyPeer(d_A2+n_half*ttRanks[1]*(long long)2, device[0], d_A22, device[2], sizeof(dt)*n_half*ttRanks[1]);
                assert(cudaStat1 == cudaSuccess);

                if(d_A22     ) cudaFree(d_A22); d_A22 = NULL;

            }
            else if(cpuid == 1)
            {      
                cudaSetDevice(device[1]);

                cudaStat2 = cudaMemcpyPeer(d_G11, device[1], d_G10, device[0], sizeof(dt)*m*ttRanks[1]);
                assert(cudaStat2 == cudaSuccess);

                smallargeMatrixMulti_slice_host_Tans(cublasH1, d_G11, h_A+((long long)m)*n_half, ttRanks[1], m, n_half, d_A21, p);
                printf("thread 1 finish0\n");
                if(d_G11     ) cudaFree(d_G11); d_G11 = NULL;

                cudaStat1 = cudaMemcpyPeer(d_A2+n_half*ttRanks[1], device[0], d_A21, device[1], sizeof(dt)*n_half*ttRanks[1]);
                assert(cudaStat1 == cudaSuccess);

                if(d_A21     ) cudaFree(d_A21); d_A21 = NULL;

            }
            else if(cpuid == 0)
            {   
                cudaSetDevice(device[0]);
                smallargeMatrixMulti_slice_host_Tans(cublasH, d_G10, h_A, ttRanks[1], m, n_half, d_A2, p);
                printf("thread 0 finish0\n");
            }
        }
        cudaStat1 = cudaDeviceSynchronize();
        assert(cudaStat1 == cudaSuccess);
        cudaSetDevice(device[0]);

        if(d_G10     ) cudaFree(d_G10); d_G10 = NULL;


    //1->2
        // GPUTimer timer1;
        // timer1.start();
        dt *d_G2=NULL, *d_A3=NULL, *h_G2=NULL;
        m = ttRanks[1] * ttDemns[1]; //4
        n = n/ttDemns[1]; //16
        cudaStat1 = cudaMalloc((void**)&d_G2, sizeof(dt)*m*ttRanks[2]);
        cudaStat2 = cudaHostAlloc((void**)&h_G2,sizeof(dt)*m*ttRanks[2],0);
        cudaStat2 = cudaMalloc((void**)&d_A3, sizeof(dt)*n*ttRanks[2]);
        assert(cudaStat1 == cudaSuccess);
        assert(cudaStat2 == cudaSuccess);
        // printMatrix_Device(10, 10, d_A2, 10, "A22");
        // printMatrix_Device(m, n, d_A2, m, "A22");
        longMatrixSVD_Eign_once_device(cublasH, cusolverH, d_A2, m, n, ttRanks[2] , d_G2, d_A3, p);
        // printMatrix_Device(10, 10, d_G2, 10, "G2");
        // printMatrix_Device(10, 10, d_A3, 10, "A3");
        // printMatrix_Device(m, ttRanks[2], d_G2, m, "G2");
        // printMatrix_Device(ttRanks[2], n, d_A3, ttRanks[2], "A3");
        if(d_A2     ) cudaFree(d_A2); d_A2 = NULL;
        cudaStat5 = cudaMemcpyAsync(h_G2, d_G2, sizeof(dt)*m*ttRanks[2], cudaMemcpyDeviceToHost,0);
        // cudaStat6 = cudaDeviceSynchronize();
        assert(cudaStat5 == cudaSuccess);
        // assert(cudaStat6 == cudaSuccess);
        if(d_G2     ) cudaFree(d_G2); d_G2 = NULL;
        // printf(" second time： %f\n\n", timer1.seconds());

    //1->2->3
        // GPUTimer timer2;
        // timer2.start();
        dt *d_G3=NULL, *d_A4=NULL, *h_G3=NULL;
        m = ttRanks[2] * ttDemns[2];
        n = n/ttDemns[2];
        cudaStat1 = cudaMalloc((void**)&d_G3, sizeof(dt)*m*ttRanks[3]);
        cudaStat2 = cudaHostAlloc((void**)&h_G3,sizeof(dt)*m*ttRanks[3],0);
        cudaStat2 = cudaMalloc((void**)&d_A4, sizeof(dt)*ttRanks[3]*n);
        assert(cudaStat1 == cudaSuccess);
        assert(cudaStat2 == cudaSuccess);
        // printMatrix_Device(10, 10, d_A3, 10, "A32");
        // printMatrix_Device(m, n, d_A3, m, "A32");
        longMatrixSVD_Eign_once_device(cublasH, cusolverH, d_A3, m, n, ttRanks[3] , d_G3, d_A4, p);
        // printMatrix_Device(10, 10, d_G3, 10, "G3");
        // printMatrix_Device(10, 10, d_A4, 10, "A4");
        // printMatrix_Device(m, ttRanks[3], d_G3, m, "G3");
        // printMatrix_Device(ttRanks[3], n, d_A4, ttRanks[3], "A4");
        if(d_A3     ) cudaFree(d_A3); d_A3 = NULL;
        cudaStat5 = cudaMemcpyAsync(h_G3, d_G3, sizeof(dt)*m*ttRanks[3], cudaMemcpyDeviceToHost,0);
        // cudaStat6 = cudaDeviceSynchronize();
        assert(cudaStat5 == cudaSuccess);
        // assert(cudaStat6 == cudaSuccess);
        if(d_G3     ) cudaFree(d_G3); d_G3 = NULL;
        // printf(" third time： %f\n\n", timer2.seconds());

    //1->2->3->4
        // GPUTimer timer3;
        // timer3.start();
        dt *d_G4=NULL, *d_A5=NULL, *h_G4=NULL;
        m = ttRanks[3] * ttDemns[3];
        n = n/ttDemns[3];
        cudaStat1 = cudaMalloc((void**)&d_G4, sizeof(dt)*m*ttRanks[4]);
        cudaStat2 = cudaHostAlloc((void**)&h_G4,sizeof(dt)*m*ttRanks[4],0);
        cudaStat2 = cudaMalloc((void**)&d_A5, sizeof(dt)*ttRanks[4]*n);
        assert(cudaStat1 == cudaSuccess);
        assert(cudaStat2 == cudaSuccess);
        // printMatrix_Device(10, 10, d_A4, 10, "A42");
        // printMatrix_Device(m, n, d_A4, m, "A42");
        longMatrixSVD_Eign_once_device(cublasH, cusolverH, d_A4, m, n, ttRanks[4] , d_G4, d_A5, p);
        // printMatrix_Device(10, 10, d_G4, 10, "G4");
        // printMatrix_Device(10, 10, d_A5, 10, "A5");
        // printMatrix_Device(m, ttRanks[4], d_G4, m, "G4");
        // printMatrix_Device(ttRanks[4], n, d_A5, ttRanks[4], "A5");
        if(d_A4     ) cudaFree(d_A4); d_A4 = NULL;
        cudaStat5 = cudaMemcpyAsync(h_G4, d_G4, sizeof(dt)*m*ttRanks[4], cudaMemcpyDeviceToHost,0);
        // cudaStat6 = cudaDeviceSynchronize();
        assert(cudaStat5 == cudaSuccess);
        // assert(cudaStat6 == cudaSuccess);
        if(d_G4     ) cudaFree(d_G4); d_G4 = NULL;
        // printf(" forth time： %f\n\n", timer3.seconds());

    //1->2->3->4->5
        // GPUTimer timer4;
        // timer4.start();
        dt *d_G5=NULL, *d_G6=NULL, *h_G5=NULL, *h_G6=NULL;
        m = ttRanks[4] * ttDemns[4];
        n = n/ttDemns[4];
        cudaStat1 = cudaMalloc((void**)&d_G5, sizeof(dt)*m*ttRanks[5]);
        cudaStat2 = cudaHostAlloc((void**)&h_G5,sizeof(dt)*m*ttRanks[5],0);
        cudaStat2 = cudaMalloc((void**)&d_G6, sizeof(dt)*ttRanks[5]*n);
        cudaStat2 = cudaHostAlloc((void**)&h_G6,sizeof(dt)*n*ttRanks[5],0);
        assert(cudaStat1 == cudaSuccess);
        assert(cudaStat2 == cudaSuccess);
        // printMatrix_Device(10, 10, d_A5, 10, "A52");
        // printMatrix_Device(m, n, d_A5, m, "A52");
        longMatrixSVD_Eign_once_device(cublasH, cusolverH, d_A5, m, n, ttRanks[5] , d_G5, d_G6, p);
        // printMatrix_Device(10, 10, d_G5, 10, "G5");
        // printMatrix_Device(10, 10, d_G6, 10, "G6");
        // printMatrix_Device(m, ttRanks[5], d_G5, m, "G5");
        // printMatrix_Device(ttRanks[5], n, d_G6, ttRanks[5], "A6");
        if(d_A5     ) cudaFree(d_A5); d_A5 = NULL;
        cudaStat5 = cudaMemcpyAsync(h_G5, d_G5, sizeof(dt)*m*ttRanks[5], cudaMemcpyDeviceToHost,0);
        cudaStat6 = cudaMemcpyAsync(h_G6, d_G6, sizeof(dt)*n*ttRanks[5], cudaMemcpyDeviceToHost,0);
        // cudaStat6 = cudaDeviceSynchronize();
        assert(cudaStat5 == cudaSuccess);
        assert(cudaStat6 == cudaSuccess);
        if(d_G5     ) cudaFree(d_G5); d_G5 = NULL;
        if(d_G6     ) cudaFree(d_G6); d_G6 = NULL;
        // printf(" fifth time： %f\n\n", timer4.seconds());

        if(calError){
            // printMatrix(10,10,h_A,10,"hA");
            double MSE = calMSE(cublasH, h_A, h_G1, h_G2, h_G3, h_G4, h_G5, h_G6, ttRanks, ttDemns, p);
            printf("\n*********************** MSE: %f ******************\n", MSE);
        }

        if(h_G1     ) cudaFreeHost(h_G1);h_G1 = NULL;
        if(h_G2     ) cudaFreeHost(h_G2);h_G2 = NULL;
        if(h_G3     ) cudaFreeHost(h_G3);h_G3 = NULL;
        if(h_G4     ) cudaFreeHost(h_G4);h_G4 = NULL;
        if(h_G5     ) cudaFreeHost(h_G5);h_G5 = NULL;
        if(h_G6     ) cudaFreeHost(h_G6);h_G6 = NULL;
        if(cublasH  ) cublasDestroy(cublasH);
        if(cusolverH) cusolverDnDestroy(cusolverH); 
    }
    float time = timer.seconds()/calculateTimes;
    return time;
}





int main(){
    ofstream ofs;
    ofs.open("result_8.txt", ios::out);
    for(int i = 8; i < 40; i=i+4){
        const int k = i;
        int calculateTimes = 10;

        bool calError = true;   
        if(calError)
            calculateTimes = 1;
        int mid_rank = 4;
        if(k == 2)
            mid_rank = 2;
        if(k == 4)
            mid_rank = 2;
        if(k > 24){
            mid_rank = 8;
        }
        if(k > 52){
            mid_rank = 8;
        }
        int ttRanks[7] = {1, mid_rank, mid_rank, mid_rank, mid_rank, mid_rank, 1};
        int ttDemns[6] = {k, k, k, k, k, k};

        int p = 8;
        if(k == 2)
            p=2;
        int m = k;
        long long n = pow(k,5);
        printf("%lld , mid_rank = %d, k = %d \n", sizeof(dt)*m*n, mid_rank,k);
        srand(1134);
        dt *h_A = NULL;
        cudaStat1 = cudaHostAlloc((void**)&h_A,sizeof(dt)*m*n,0);
        assert(cudaStat1 == cudaSuccess);
        generalTTTensor(h_A, ttRanks, ttDemns, p);
        // printMatrix(10,10,h_A,10,"A");
        warmupcu();
        // ttdec_half_lu_host_6_multiGPU(h_A, ttRanks, ttDemns, 1, p, calError);
        float time = 0;
        if(k<40){
            printf("device\n");
            // ttdec_half_lu_device_6_multiGPU(h_A, ttRanks, ttDemns, 1, p, calError);
            time = ttdec_half_lu_device_6_multiGPU(h_A, ttRanks, ttDemns, calculateTimes, p, calError);
        }
        else{
            printf("host\n");
            // ttdec_half_lu_host_6_multiGPU(h_A, ttRanks, ttDemns, 1, p, calError);
            time = ttdec_half_lu_host_6_multiGPU(h_A, ttRanks, ttDemns, calculateTimes, p, calError);
        }

        printf("*************time****************\n %f \n*******************************\n", time);
        ofs << "Size： " << i << " CPU Time:  " << time << endl;
        if(h_A      ) cudaFreeHost(h_A); h_A = NULL;  
    }
    ofs.close();
    return 0;
}
