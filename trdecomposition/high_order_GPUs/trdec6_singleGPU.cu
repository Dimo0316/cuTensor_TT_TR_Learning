#include <iostream>
#include <fstream>
#include <assert.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <math.h>
#include <curand.h>
#include <cufft.h>
#include <cuda_fp16.h>

// /usr/local/cuda/bin/nvcc -lcublas -lcusolver -lcurand -std=c++11 trdec6_oneGPU.cu -o oneGPU
// G2*reshape(reshape(G3,[4,2])*reshape(reshape(G4,[4,2]) * reshape(G5*G6,[2,8]),[2,16]),[2,32])
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
__global__ void tensorToMode2(dt *T1,dt *T2,int m,int n,int k){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<m*n*k){
        int tube = i/(m*n);
        int row = (i-tube*(m*n))%m;
        int col = (i-tube*(m*n))/m;
        T2[tube*m*n+row*n+col] = T1[tube*m*n+col*m+row];
    }
    __syncthreads();
}

__global__ void tensorToMode231(dt *T1,dt *T2,int m,int n,int k){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<m*n*k){
        int tube = i/(m*n);
        int row = (i-tube*(m*n))%m;
        int col = (i-tube*(m*n))/m;
        T2[row*k*n+tube*n+col] = T1[tube*m*n+col*m+row]; //T1[i]
    }
    __syncthreads();
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

double norm2HH(float *A, long long len){
  printf("%lld\n", len);
  double norm2Result = 0.0;
  for(long long i = 0; i < len; ++i){
    norm2Result += (A[i] * A[i]);
    if(i%10000000000 == 0)
        printf("%f\n", norm2Result);
  }
  norm2Result = sqrtf(norm2Result);
  return norm2Result;
}

void tensorToMode3_host(dt *T1, dt *T2 , long long m, long long n, long long k){
    for(long long i = 0; i < m*n*k; ++i){
        long long tube = i/(m*n);
        long long row = (i-tube*(m*n))%m;
        long long col = (i-tube*(m*n))/m;
        T2[k*col*m+row*k+tube] = T1[tube*m*n+col*m+row];
    }
}

double calMSE(cublasHandle_t cublasH, dt *h_A, dt *h_G1, dt *h_G2, dt *h_G3, dt *h_G4, dt *h_G5, dt *h_G6, const int *ttRanks, const int *ttDemns, const int p){
    long long calNumber1 = ttDemns[0]*ttDemns[1]*ttDemns[2];
    long long calNumber = calNumber1*(long long)ttDemns[3]*ttDemns[4]*ttDemns[5];
    printf("calNumber : %lld\n", calNumber);
    double normA = norm2HH(h_A, calNumber);
    cout<<"normA: "<<normA<<endl;
    dt *h_G56=NULL, *h_G456=NULL, *h_G3456=NULL, *h_G23456=NULL, *h_G123456=NULL, *h_G23456_mid=NULL;
    cudaStat3 = cudaMallocHost((void**)&h_G56, sizeof(dt)*ttRanks[4] *ttDemns[4]*ttDemns[5]* ttRanks[6]);
    assert(cudaStat3 == cudaSuccess);
    smallargeMatrixMulti_slice_host(cublasH, h_G5, h_G6, ttRanks[4]*ttDemns[4], ttRanks[5], ttDemns[5]*ttRanks[6], h_G56, 1, beta0);
    // printMatrix(ttRanks[4]*ttDemns[4],ttRanks[5],h_G5,ttRanks[4]*ttDemns[4],"h5");
    // printMatrix(ttRanks[5],ttDemns[5]*ttRanks[6],h_G6,ttRanks[5],"h6");
    // printMatrix(ttRanks[4]*ttDemns[4],ttDemns[5]*ttRanks[6],h_G56,ttRanks[4]*ttDemns[4],"h561");

    cudaStat3 = cudaMallocHost((void**)&h_G456, sizeof(dt)*ttRanks[3] *ttDemns[3]*ttDemns[4]*ttDemns[5]* ttRanks[6]);
    assert(cudaStat3 ==cudaSuccess);
    smallargeMatrixMulti_slice_host(cublasH, h_G4, h_G56, ttRanks[3]*ttDemns[3], ttRanks[4], ttDemns[4]*ttDemns[5]*ttRanks[6], h_G456, 1, beta0);
    // printMatrix(ttRanks[3]*ttDemns[3],ttRanks[4],h_G4,ttRanks[3]*ttDemns[3],"h4");
    // printMatrix(ttRanks[4],ttDemns[4]*ttDemns[5]*ttRanks[6],h_G56,ttRanks[4],"h56");
    // printMatrix(ttRanks[3]*ttDemns[3],ttDemns[4]*ttDemns[5]*ttRanks[6],h_G456,ttRanks[3]*ttDemns[3],"h4561");
    if(h_G56) cudaFreeHost(h_G56); h_G56 = NULL;

    cudaStat3 = cudaMallocHost((void**)&h_G3456, sizeof(dt)*ttRanks[2]*ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5]* ttRanks[6]);
    assert(cudaStat3 == cudaSuccess);
    smallargeMatrixMulti_slice_host(cublasH, h_G3, h_G456, ttRanks[2]*ttDemns[2], ttRanks[3], ttDemns[3]*ttDemns[4]*ttDemns[5]*ttRanks[6], h_G3456, 1, beta0);
    // printMatrix(ttRanks[2]*ttDemns[2],ttRanks[3],h_G3,ttRanks[2]*ttDemns[3],"h3");
    // printMatrix(ttRanks[3],ttDemns[3]*ttDemns[4]*ttDemns[5]*ttRanks[6],h_G456,ttRanks[3],"h456");
    // printMatrix(ttRanks[2]*ttDemns[2],ttDemns[3]*ttDemns[4]*ttDemns[5]*ttRanks[6],h_G3456,ttRanks[2]*ttDemns[2],"h34561");
    if(h_G456) cudaFreeHost(h_G456); h_G456 = NULL;

    cudaStat3 = cudaMallocHost((void**)&h_G23456, sizeof(dt)*ttRanks[1]*ttDemns[1]*ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5]*ttRanks[6]);
    assert(cudaStat3 == cudaSuccess);
    smallargeMatrixMulti_slice_host(cublasH, h_G2, h_G3456, ttRanks[1]*ttDemns[1], ttRanks[2], ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5]*ttRanks[6], h_G23456, 4, beta0);
    // printMatrix(ttRanks[1]*ttDemns[1],ttRanks[2],h_G2,ttRanks[1]*ttDemns[1],"h2");
    // printMatrix(ttRanks[2],ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5]*ttRanks[6],h_G3456,ttRanks[2],"h3456");
    // printMatrix(ttRanks[1]*ttDemns[1],ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5]*ttRanks[6],h_G23456,ttRanks[1]*ttDemns[1],"h234561");
    if(h_G3456) cudaFreeHost(h_G3456); h_G3456 = NULL;

///TODO 将TR中的G1 和 G23456进行变换
    cudaStat3 = cudaMallocHost((void**)&h_G23456_mid, sizeof(dt)*ttRanks[1]*ttDemns[1]*ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5]*ttRanks[6]);
    tensorToMode3_host(h_G23456, h_G23456_mid, ttRanks[1], ttDemns[1]*ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5], ttRanks[6]);
    if(h_G23456) cudaFreeHost(h_G23456); h_G23456 = NULL;

    smallargeMatrixMulti_slice_host(cublasH, h_G1, h_G23456_mid, ttDemns[0], ttRanks[0]*ttRanks[1], ttDemns[1]*ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5], h_A, p, beta_1);
    // printMatrix(ttRanks[0]*ttDemns[0],ttRanks[1],h_G1,ttRanks[0]*ttDemns[0],"h1");
    // printMatrix(ttRanks[1],ttDemns[1]*ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5]*ttRanks[6],h_G23456,ttRanks[2],"h23456");
    // printMatrix(ttDemns[0],ttDemns[1]*ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5],h_A,ttDemns[0],"h123456");   
    if(h_G23456_mid) cudaFreeHost(h_G23456_mid);

    double normAr = norm2HH(h_A, calNumber);
    cout<<"normAr: "<<normAr<<endl;
    double MSE = normAr/normA;
    return MSE;
}


void generalTRTensor(dt *h_A, const int *ttRanks, const int *ttDemns, const int &p){
    dt *h_G56=NULL, *h_G456=NULL, *h_G3456=NULL, *h_G23456=NULL, *h_G123456=NULL, *h_G23456_mid=NULL;
    dt *h_G1=NULL,  *h_G2=NULL,  *h_G3=NULL,  *h_G4=NULL,  *h_G5=NULL,  *h_G6=NULL;
    cublasHandle_t cublasH = NULL;
    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    cublas_status = cublasSetMathMode(cublasH, mathMode);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    cudaStat1 = cudaMallocHost((void**)&h_G5, sizeof(dt)*ttRanks[4]*ttDemns[4]*ttRanks[5]);
    cudaStat2 = cudaMallocHost((void**)&h_G6, sizeof(dt)*ttRanks[5]*ttDemns[5]*ttRanks[6]);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    srand(1134);
    for(long long i = 0; i < ttRanks[4]*ttDemns[4]*ttRanks[5]; i++){
        // h_G5[i] = (dt) ((rand()*1.0) / (RAND_MAX*1.0))*2.0 - 1.0;
        h_G5[i] = (dt) ((rand()*1.0) / (RAND_MAX*1.0))*2.0 - 1.0;
    }
    for(long long i = 0; i < ttRanks[5]*ttDemns[5]*ttRanks[6]; i++){
        h_G6[i] = (dt) ((rand()*1.0) / (RAND_MAX*1.0))*2.0 - 1.0;
    }
    
    cudaStat3 = cudaMallocHost((void**)&h_G56, sizeof(dt)*ttRanks[4] *ttDemns[4]*ttDemns[5]* ttRanks[6]);
    assert(cudaStat3 == cudaSuccess);
    smallargeMatrixMulti_slice_host(cublasH, h_G5, h_G6, ttRanks[4]*ttDemns[4], ttRanks[5], ttDemns[5]*ttRanks[6], h_G56, 1, beta0);
    // printMatrix(ttRanks[4]*ttDemns[4],ttRanks[5],h_G5,ttRanks[4]*ttDemns[4],"h5");
    // printMatrix(ttRanks[5],ttDemns[5]*ttRanks[6],h_G6,ttRanks[5],"h6");
    // printMatrix(ttRanks[4]*ttDemns[4],ttDemns[5]*ttRanks[6],h_G56,ttRanks[4]*ttDemns[4],"h561");
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
    // printMatrix(ttRanks[3]*ttDemns[3],ttRanks[4],h_G4,ttRanks[3]*ttDemns[3],"h4");
    // printMatrix(ttRanks[4],ttDemns[4]*ttDemns[5]*ttRanks[6],h_G56,ttRanks[4],"h56");
    // printMatrix(ttRanks[3]*ttDemns[3],ttDemns[4]*ttDemns[5]*ttRanks[6],h_G456,ttRanks[3]*ttDemns[3],"h4561");
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
    // printMatrix(ttRanks[2]*ttDemns[2],ttRanks[3],h_G3,ttRanks[2]*ttDemns[3],"h3");
    // printMatrix(ttRanks[3],ttDemns[3]*ttDemns[4]*ttDemns[5]*ttRanks[6],h_G456,ttRanks[3],"h456");
    // printMatrix(ttRanks[2]*ttDemns[2],ttDemns[3]*ttDemns[4]*ttDemns[5]*ttRanks[6],h_G3456,ttRanks[2]*ttDemns[2],"h34561");    

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
    // printMatrix(ttRanks[1]*ttDemns[1],ttRanks[2],h_G2,ttRanks[1]*ttDemns[1],"h2");
    // printMatrix(ttRanks[2],ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5]*ttRanks[6],h_G3456,ttRanks[2],"h3456");
    // printMatrix(ttRanks[1]*ttDemns[1],ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5]*ttRanks[6],h_G23456,ttRanks[1]*ttDemns[1],"h234561");   

    if(h_G3456) cudaFreeHost(h_G3456); h_G3456 =NULL;
    if(h_G2) cudaFreeHost(h_G2);h_G2=NULL;
///TODO 让G23456 变换形式
    cudaStat3 = cudaMallocHost((void**)&h_G23456_mid, sizeof(dt)*ttRanks[1]*ttDemns[1]*ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5]*ttRanks[6]);
    assert(cudaStat3 == cudaSuccess);

    // printMatrix(ttRanks[1],ttDemns[1]*ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5]*ttRanks[6],h_G23456,ttRanks[1],"h23456");
    tensorToMode3_host(h_G23456, h_G23456_mid, ttRanks[1], ttDemns[1]*ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5], ttRanks[6]);
    // printMatrix(ttRanks[1]*ttRanks[6],ttDemns[1]*ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5],h_G23456_mid, ttRanks[1]*ttRanks[6],"h23456to3");
    if(h_G23456) cudaFreeHost(h_G23456); h_G23456 =NULL;


    cudaStat1 = cudaMallocHost((void**)&h_G1, sizeof(dt)*ttRanks[0]*ttDemns[0]*ttRanks[1]);
    assert(cudaStat1 == cudaSuccess);
    for(long long i = 0; i < ttRanks[0]*ttDemns[0]*ttRanks[1]; i++){
        h_G1[i] = (dt) ((rand()*1.0) / (RAND_MAX*1.0))*2.0 - 1.0;
    }

    smallargeMatrixMulti_slice_host(cublasH, h_G1, h_G23456_mid, ttDemns[0], ttRanks[0]*ttRanks[1], ttDemns[1]*ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5], h_A, p, beta0);
    // printMatrix(ttDemns[0],ttRanks[0]*ttRanks[1], h_G1, ttDemns[0],"h1");
    // printMatrix(ttRanks[1]*ttRanks[6],ttDemns[1]*ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5],h_G23456_mid, ttRanks[1]*ttRanks[6],"h23456to3");
    // printMatrix(ttDemns[0],ttDemns[1]*ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5],h_A,ttDemns[0],"hA0");   
    // printMatrix(m, n, h_A, m, "hA1");
    if(h_G23456_mid) cudaFreeHost(h_G23456_mid); h_G23456_mid =NULL;
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



// nvcc -lcublas -lcusolver -lcurand -std=c++11 svd.cu -o svd
void printMatrix_Device(int m, int n, const dt*d_A, int lda, const char* name)
{
    dt *h_A = NULL;
    cudaHostAlloc((void**)&h_A , sizeof(dt)*m*n, 1);
    cudaMemcpy(h_A, d_A, sizeof(dt)*m*n, cudaMemcpyDeviceToHost);
    printMatrix(m, n, h_A, lda, name);
    if(h_A) cudaFreeHost(h_A);
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

    cudaMalloc((void**)&devInfo, sizeof(int));
    
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
}

// large matrix self multiply A*AT
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

// small matrix multiply big matrix AT*B
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

void smallargeMatrixMulti_once_device_float(cublasHandle_t cublasH, dt *d_A, dt *d_B, const int &ttrank, const int &m, const long long &n, dt *d_ATB){
   
    cublas_status = cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                           ttrank, n, m,
                           &alpha, d_A,  m,
                           d_B,  m,
                           &beta0, d_ATB,  ttrank);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    
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

void smallargeMatrixMulti_slice_device_float(cublasHandle_t cublasH, dt *d_A, dt *d_B, const int &ttrank, const int &m, const long long &n, dt *d_ATB, const int &p){
    long long slice = n/p;
   
    for(int i = 0; i<p; i++){
        cublas_status = cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                           ttrank, slice, m,
                           &alpha, d_A,  m,
                           d_B+i*m*slice, m,
                           &beta0, d_ATB+slice*i*ttrank, ttrank);

     
        assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    }
}

void longMatrixSVD_Eign_once_device(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, dt *d_A, const int &m, const long long &n, const int &ttRank, dt *d_G, dt *d_A2, const int &p){
    dt *d_W=NULL, *d_AAT=NULL;
    cudaStat1 = cudaMalloc((void**)&d_AAT, sizeof(dt)*m*m);
    cudaStat2 = cudaMalloc((void**)&d_W, sizeof(dt)*m);
    // printf("largeMatrixSelfMulti_once_device\n");
    largeMatrixSelfMulti_once_device(cublasH, d_A, m, n, d_AAT);

    // printf("basicEig\n");
    basicEig(cusolverH, d_AAT, m, d_W);
    // printf("basicEig Finish\n");
    if(d_W     ) cudaFree(d_W); d_W = NULL;
    // printf("matrixInvertColumn\n");
    matrixInvertColumn(d_AAT, d_G, m, ttRank);
    if(d_AAT    ) cudaFree(d_AAT);d_AAT=NULL;
    // printMatrix_Device( 10,10, d_G1, 10, "G1");
    // 大于24需要分片 
    if(m>24 && n>pow(m,4)){
        // printf("smallargeMatrixMulti_slice_device\n");
        smallargeMatrixMulti_slice_device(cublasH, d_G, d_A, ttRank, m, n, d_A2, p);
        // printMatrix_Device(10, 10, d_A2, 10, "A2");
    }
    else{
        // printf("smallargeMatrixMulti_once_device\n");
        smallargeMatrixMulti_once_device(cublasH, d_G, d_A, ttRank, m, n, d_A2);
    }
    // printMatrix_Device(10, 10, d_A2, 10, "A2");
}


// 实现一个超大规模矩阵A的 A*AT，存储到GPU上
void largeMatrixSelfMulti_slice_host(cublasHandle_t cublasH, dt *h_A,const int &m,const long long &n, dt *d_AAT,const int &p){
    long long slice = 0, le = 0;
    if(n%p==0){
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
    // printf("%d\n", slice);
    for(int i=0;i<p;i++){
        cudaStat3 = cudaMemcpyAsync(d_tempA, h_A+m*slice*i, sizeof(dt)*m*slice, cudaMemcpyHostToDevice,0);
        cudaStat4 = cudaDeviceSynchronize();
        assert(cudaStat3 == cudaSuccess);
        assert(cudaStat4 == cudaSuccess);
        //判断是否降维
        // printMatrix_Device(m, slice, d_tempA, m, "d_tempA");
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

void smallargeMatrixMulti_slice_host_Tans(cublasHandle_t cublasH, dt *d_A, dt *h_B, const int &ttrank, const int &m,  const long long &n, dt *d_ATB, const int &p){
    long long slice = 0;
    if(n%p==0)
        slice = n/p;
    else
        assert("P is not slice");
    // printf("ttrank: %d, n: %d",m,ttrank,n);
    float *d_tempB = NULL;//, *d_tempATB = NULL; 
    half *d_Ah = NULL, *d_Bh = NULL;
    cudaStat1 = cudaMalloc((void**)&d_tempB, sizeof(dt)*m*slice);
    // cudaStat2 = cudaMalloc((void**)&d_tempATB, sizeof(dt)*ttrank*slice);
    cudaStat3 = cudaMalloc((void**)&d_Ah, sizeof(half)*m*ttrank);
    cudaStat4 = cudaMalloc((void**)&d_Bh, sizeof(half)*m*slice);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);
    assert(cudaStat4 == cudaSuccess);

    // printMatrix_Device(m, ttrank, d_A, m, "tempA");
    f2h(d_A, d_Ah, m*ttrank);
    // printf("%d\n", slice);
    for(int i=0;i<p;i++){
        cudaStat3 = cudaMemcpyAsync(d_tempB, h_B+m*slice*i, sizeof(dt)*m*slice, cudaMemcpyHostToDevice,0);
        cudaStat4 = cudaDeviceSynchronize();
        assert(cudaStat3 == cudaSuccess);
        assert(cudaStat4 == cudaSuccess);
        // printMatrix_Device(m, slice, d_tempB, m, "tempB");
        f2h(d_tempB, d_Bh, m*slice);
        cublas_status = cublasGemmEx(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                           ttrank, slice, m,
                           &alpha, d_Ah, CUDA_R_16F, m,
                           d_Bh, CUDA_R_16F, m,
                           &beta0, d_ATB+ttrank*slice*i, CUDA_R_32F, ttrank,
                           CUDA_R_32F, algo);
        assert(cublas_status == CUBLAS_STATUS_SUCCESS);
        // printMatrix_Device(ttrank, slice, d_tempATB, ttrank, "tempATB");

        // cudaStat3 = cudaMemcpyAsync(h_ATB+ttrank*slice*i, d_tempATB, sizeof(dt)*ttrank*slice, cudaMemcpyDeviceToHost, 0);
        cudaStat4 = cudaDeviceSynchronize();
        // assert(cudaStat3 == cudaSuccess);
        assert(cudaStat4 == cudaSuccess);
    }
    if(d_Ah     ) cudaFree(d_Ah); d_Ah=NULL;
    if(d_Bh     ) cudaFree(d_Bh); d_Bh=NULL;
    if(d_tempB  ) cudaFree(d_tempB); d_tempB=NULL;
    // if(d_tempATB  ) cudaFree(d_tempATB); d_tempATB=NULL;
}


void longMatrixSVD_Eign_once_host(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, dt *h_A, const int &m, const long long &n, const int &ttRank, dt *h_G, dt *d_A2, const int &p){
    dt *d_W=NULL, *d_AAT=NULL,*d_G=NULL;
    cudaStat1 = cudaMalloc((void**)&d_AAT, sizeof(dt)*m*m);
    cudaStat2 = cudaMalloc((void**)&d_W, sizeof(dt)*m);
    cudaStat3 = cudaMalloc((void**)&d_G, sizeof(dt)*m*ttRank);

    // printf("selfMtrixMul\n");
    // printMatrix_Device( m, n, d_A, m, "A");
    largeMatrixSelfMulti_slice_host(cublasH, h_A, m, n, d_AAT, p);
    // printMatrix_Device( m, m, d_AAT, m, "AAT");
    // printf("basicEig\n");
    basicEig(cusolverH, d_AAT, m, d_W);
     // printMatrix_Device( m, m, d_AAT, m, "AAT");
    if(d_W     ) cudaFree(d_W); d_W = NULL;
    // printf("matrixInvertColumn\n");
    matrixInvertColumn(d_AAT, d_G, m, ttRank);
    if(d_AAT    ) cudaFree(d_AAT);d_AAT=NULL;
    // printMatrix_Device( 10,10, d_G1, 10, "G1");
    // 大于24需要分片

    // printf("smallargeMatrixMulti_slice_host_Tans\n");
    smallargeMatrixMulti_slice_host_Tans(cublasH, d_G, h_A, ttRank, m, n, d_A2, p);
    // printMatrix_Device(10, 10, d_A2, 10, "A2");

    cudaStat1 = cudaMemcpyAsync(h_G, d_G, sizeof(dt)*m*ttRank, cudaMemcpyDeviceToHost,0);
    assert(cudaStat1 == cudaSuccess);
    if(d_G) cudaFree(d_G);d_G=NULL;
    // printMatrix_Device(10, 10, d_A2, 10, "A2");
}



float ttdec_half_lu_host_6(dt* h_A, int *ttRanks, const int *ttDemns, const int &calculateTimes, const int &p, const bool calError){
    printf("Start mul \n");
    GPUTimer timer;
    timer.start();
    float timeTransfer = 0.0;
    for(int i=0;i<calculateTimes;i++){
        printf("***************************** %d ********************************\n", i);
        cublasHandle_t cublasH = NULL;
        cusolverDnHandle_t cusolverH = NULL;

        cusolver_status = cusolverDnCreate(&cusolverH);
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

        cublas_status = cublasCreate(&cublasH);
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);

        cublas_status = cublasSetMathMode(cublasH, mathMode);
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);
        
    //1
        int m = ttDemns[0];
        int n = ttDemns[1]*ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5];

        dt *h_G1=NULL;
        dt *d_A2=NULL;
        dt *d_A2_mid=NULL;
        cudaStat1 = cudaMalloc((void**)&d_A2_mid, sizeof(dt)*n*ttRanks[1]*ttRanks[0]);
        cudaStat2 = cudaMalloc((void**)&d_A2, sizeof(dt)*n*ttRanks[1]*ttRanks[0]);
        cudaStat3 = cudaHostAlloc((void**)&h_G1,sizeof(dt)*m*ttRanks[1]*ttRanks[0],0);
        // cudaStat1 = cudaHostAlloc((void**)&h_A1,sizeof(dt)*n*ttRanks[1],0);
        assert(cudaStat1 == cudaSuccess);
        assert(cudaStat2 == cudaSuccess);
        assert(cudaStat3 == cudaSuccess);
        // printMatrix_Device(m, n, d_A, m, "A");
        longMatrixSVD_Eign_once_host(cublasH, cusolverH, h_A, m, n, ttRanks[1]*ttRanks[0], h_G1, d_A2_mid, p);


        GPUTimer timer0;
        timer0.start();
    //将张量转成231的形式
        dim3 threads(1024,1,1);
        dim3 blocksr1r2n((ttRanks[6]*ttRanks[1]*n+1024-1)/1024,1,1);
        tensorToMode231<<<blocksr1r2n, threads>>>(d_A2_mid, d_A2, ttRanks[6], ttRanks[1], n);
        // printMatrix_Device(ttRanks[1]*ttRanks[6], n, d_A2temp, ttRanks[1]*ttRanks[6], "A2temp");
        cudaStat1 = cudaDeviceSynchronize();
        if(d_A2_mid     ) cudaFree(d_A2_mid); d_A2_mid = NULL;
        assert(cudaStat1 == cudaSuccess);
        timeTransfer += timer0.seconds();
        // printMatrix(10, 10, h_G1, 10, "G1");
        // printMatrix(10, 10, h_A2, 10, "A2");

        // cudaStat2 = cudaMemcpyAsync(d_A2, h_A1, sizeof(dt)*n*ttRanks[1], cudaMemcpyHostToDevice,0);
        // cudaStat3 = cudaDeviceSynchronize();
        // assert(cudaStat1 == cudaSuccess);
        // assert(cudaStat2 == cudaSuccess);
        // if(h_A1     ) cudaFreeHost(h_A1);h_A1 = NULL;

    //1->2
        dt *d_G2=NULL, *d_A3=NULL, *h_G2=NULL;
        m = ttRanks[1] * ttDemns[1];
        n = n/ttDemns[1] * ttRanks[6];
        cudaStat1 = cudaMalloc((void**)&d_G2, sizeof(dt)*m*ttRanks[2]);
        cudaStat2 = cudaHostAlloc((void**)&h_G2,sizeof(dt)*m*ttRanks[2],0);
        cudaStat2 = cudaMalloc((void**)&d_A3, sizeof(dt)*n*ttRanks[2]);
        assert(cudaStat1 == cudaSuccess);
        assert(cudaStat2 == cudaSuccess);
        // printMatrix_Device(ttRanks[1]*m, n, d_A2, ttRanks[1]*m, "A22");
        longMatrixSVD_Eign_once_device(cublasH, cusolverH, d_A2, m, n, ttRanks[2] , d_G2, d_A3, p);
        // printMatrix_Device(ttRanks[1]*m, ttRanks[2], d_G2, ttRanks[1]*m, "G22");
        // printMatrix_Device(ttRanks[2], n, d_A3, ttRanks[2], "A32");
        if(d_A2     ) cudaFree(d_A2); d_A2 = NULL;
        cudaStat5 = cudaMemcpyAsync(h_G2, d_G2, sizeof(dt)*m*ttRanks[2], cudaMemcpyDeviceToHost,0);
        // cudaStat6 = cudaDeviceSynchronize();
        assert(cudaStat5 == cudaSuccess);
        // assert(cudaStat6 == cudaSuccess);
        if(d_G2     ) cudaFree(d_G2); d_G2 = NULL;

    //1->2->3
        dt *d_G3=NULL, *d_A4=NULL, *h_G3=NULL;
        m = ttRanks[2] * ttDemns[2];
        n = n/ttDemns[2];
        cudaStat1 = cudaMalloc((void**)&d_G3, sizeof(dt)*m*ttRanks[3]);
        cudaStat2 = cudaHostAlloc((void**)&h_G3,sizeof(dt)*m*ttRanks[3],0);
        cudaStat2 = cudaMalloc((void**)&d_A4, sizeof(dt)*ttRanks[3]*n);
        assert(cudaStat1 == cudaSuccess);
        assert(cudaStat2 == cudaSuccess);
        // printMatrix_Device(ttRanks[1]*m, n, d_A2, ttRanks[1]*m, "A22");
        longMatrixSVD_Eign_once_device(cublasH, cusolverH, d_A3, m, n, ttRanks[3] , d_G3, d_A4, p);
        // printMatrix_Device(ttRanks[1]*m, ttRanks[2], d_G2, ttRanks[1]*m, "G22");
        // printMatrix_Device(ttRanks[2], n, d_A3, ttRanks[2], "A32");
        if(d_A3     ) cudaFree(d_A3); d_A3 = NULL;
        cudaStat5 = cudaMemcpyAsync(h_G3, d_G3, sizeof(dt)*m*ttRanks[3], cudaMemcpyDeviceToHost,0);
        // cudaStat6 = cudaDeviceSynchronize();
        assert(cudaStat5 == cudaSuccess);
        // assert(cudaStat6 == cudaSuccess);
        if(d_G3     ) cudaFree(d_G3); d_G3 = NULL;

    //1->2->3->4
        dt *d_G4=NULL, *d_A5=NULL, *h_G4=NULL;
        m = ttRanks[3] * ttDemns[3];
        n = n/ttDemns[3];
        cudaStat1 = cudaMalloc((void**)&d_G4, sizeof(dt)*m*ttRanks[4]);
        cudaStat2 = cudaHostAlloc((void**)&h_G4,sizeof(dt)*m*ttRanks[4],0);
        cudaStat2 = cudaMalloc((void**)&d_A5, sizeof(dt)*ttRanks[4]*n);
        assert(cudaStat1 == cudaSuccess);
        assert(cudaStat2 == cudaSuccess);
        // printMatrix_Device(ttRanks[1]*m, n, d_A2, ttRanks[1]*m, "A22");
        longMatrixSVD_Eign_once_device(cublasH, cusolverH, d_A4, m, n, ttRanks[4] , d_G4, d_A5, p);
        // printMatrix_Device(ttRanks[1]*m, ttRanks[2], d_G2, ttRanks[1]*m, "G22");
        // printMatrix_Device(ttRanks[2], n, d_A3, ttRanks[2], "A32");
        if(d_A4     ) cudaFree(d_A4); d_A4 = NULL;
        cudaStat5 = cudaMemcpyAsync(h_G4, d_G4, sizeof(dt)*m*ttRanks[4], cudaMemcpyDeviceToHost,0);
        // cudaStat6 = cudaDeviceSynchronize();
        assert(cudaStat5 == cudaSuccess);
        // assert(cudaStat6 == cudaSuccess);
        if(d_G4     ) cudaFree(d_G4); d_G4 = NULL;

    //1->2->3->4->5
        dt *d_G5=NULL, *d_G6=NULL, *h_G5=NULL, *h_G6=NULL;
        m = ttRanks[4] * ttDemns[4];
        n = n/ttDemns[4];
        cudaStat1 = cudaMalloc((void**)&d_G5, sizeof(dt)*m*ttRanks[5]);
        cudaStat2 = cudaHostAlloc((void**)&h_G5,sizeof(dt)*m*ttRanks[5],0);
        cudaStat2 = cudaMalloc((void**)&d_G6, sizeof(dt)*ttRanks[5]*n);
        cudaStat2 = cudaHostAlloc((void**)&h_G6,sizeof(dt)*n*ttRanks[5],0);
        assert(cudaStat1 == cudaSuccess);
        assert(cudaStat2 == cudaSuccess);
        // printMatrix_Device(ttRanks[1]*m, n, d_A2, ttRanks[1]*m, "A22");
        longMatrixSVD_Eign_once_device(cublasH, cusolverH, d_A5, m, n, ttRanks[5] , d_G5, d_G6, p);
        // printMatrix_Device(ttRanks[1]*m, ttRanks[2], d_G2, ttRanks[1]*m, "G22");
        // printMatrix_Device(ttRanks[5], n, d_G6, ttRanks[5], "G6");
        if(d_A5     ) cudaFree(d_A5); d_A5 = NULL;
        cudaStat5 = cudaMemcpyAsync(h_G5, d_G5, sizeof(dt)*m*ttRanks[5], cudaMemcpyDeviceToHost,0);
        cudaStat6 = cudaMemcpyAsync(h_G6, d_G6, sizeof(dt)*n*ttRanks[5], cudaMemcpyDeviceToHost,0);
        // cudaStat6 = cudaDeviceSynchronize();
        assert(cudaStat5 == cudaSuccess);
        assert(cudaStat6 == cudaSuccess);
        if(d_G5     ) cudaFree(d_G5); d_G5 = NULL;
        if(d_G6     ) cudaFree(d_G6); d_G6 = NULL;

        if(calError){
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


float ttdec_half_lu_device_6(dt* h_A, const int *ttRanks, const int *ttDemns, const int &calculateTimes, const int &p, bool calError){
    printf("Start mul \n");
    GPUTimer timer;
    timer.start();
    float timeTransfer = 0.0;
    for(int i=0;i<calculateTimes;i++){
        printf("***************************** %d ********************************\n", i);
        cublasHandle_t cublasH = NULL;
        cusolverDnHandle_t cusolverH = NULL;

        cusolver_status = cusolverDnCreate(&cusolverH);
        assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

        cublas_status = cublasCreate(&cublasH);
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);

        cublas_status = cublasSetMathMode(cublasH, mathMode);
        assert(CUBLAS_STATUS_SUCCESS == cublas_status);

        int m = ttDemns[0] ;
        int n = ttDemns[1]*ttDemns[2]*ttDemns[3]*ttDemns[4]*ttDemns[5];
        // printMatrix(m, n, h_A, m, "hA");
        dt *d_A=NULL, *d_G1=NULL, *d_A2=NULL, *h_G1=NULL, *d_A2temp=NULL;
        cudaStat1 = cudaMalloc((void**)&d_A, sizeof(dt)*m*n);
        cudaStat2 = cudaHostAlloc((void**)&h_G1,sizeof(dt)*ttRanks[0]*m*ttRanks[1],0);
        cudaStat3 = cudaMalloc((void**)&d_G1, sizeof(dt)*ttRanks[0]*m*ttRanks[1]);
        cudaStat4 = cudaMalloc((void**)&d_A2, sizeof(dt)*ttRanks[6]*n*ttRanks[1]);
        cudaStat4 = cudaMalloc((void**)&d_A2temp, sizeof(dt)*ttRanks[6]*n*ttRanks[1]);
        cudaStat5 = cudaMemcpyAsync(d_A, h_A, sizeof(dt)*m*n, cudaMemcpyHostToDevice,0);
        cudaStat6 = cudaDeviceSynchronize();
        assert(cudaStat1 == cudaSuccess);
        assert(cudaStat2 == cudaSuccess);
        assert(cudaStat3 == cudaSuccess);
        assert(cudaStat4 == cudaSuccess);
        assert(cudaStat5 == cudaSuccess);
        assert(cudaStat6 == cudaSuccess);
        // printMatrix_Device(m, n, d_A, m, "A");
        // printMatrix_Device(10, 10, d_A, 10, "A");
        longMatrixSVD_Eign_once_device(cublasH, cusolverH, d_A, m, n, ttRanks[6]*ttRanks[1] , d_G1, d_A2temp, p);
        if(d_A      ) cudaFree(d_A); d_A = NULL;

        GPUTimer timer0;
        timer0.start();
    //将张量转成231的形式
        dim3 threads(1024,1,1);
        dim3 blocksr1r2n((ttRanks[6]*ttRanks[1]*n+1024-1)/1024,1,1);
        tensorToMode231<<<blocksr1r2n, threads>>>(d_A2temp, d_A2, ttRanks[6], ttRanks[1], n);
        // printMatrix_Device(ttRanks[1]*ttRanks[6], n, d_A2temp, ttRanks[1]*ttRanks[6], "A2temp");
        cudaStat1 = cudaDeviceSynchronize();
        assert(cudaStat1 == cudaSuccess);
        timeTransfer += timer0.seconds();

        if(d_A2temp      ) cudaFree(d_A2temp); d_A2temp = NULL;
        // printMatrix_Device(m, ttRanks[1]*ttRanks[0], d_G1, m, "G1");
        // printMatrix_Device(ttRanks[1]*m, n/m*ttRanks[6], d_A2, ttRanks[1]*m, "A2");
        // printMatrix_Device(m, ttRanks[1], d_G1, m, "G1");
        // printMatrix_Device(10, 10, d_A2, 10, "A2");
        cudaStat5 = cudaMemcpyAsync(h_G1, d_G1, sizeof(dt)*ttRanks[0]*m*ttRanks[1], cudaMemcpyDeviceToHost,0);
        // cudaStat6 = cudaDeviceSynchronize();
        assert(cudaStat5 == cudaSuccess);
        // assert(cudaStat6 == cudaSuccess);
        if(d_G1     ) cudaFree(d_G1); d_G1 = NULL;
       
///检查到这里没问题
    //1->2
        dt *d_G2=NULL, *d_A3=NULL, *h_G2=NULL;
        m = ttRanks[1] * ttDemns[1]; //4
        n = n/ttDemns[1] * ttRanks[6]; //16
        cudaStat1 = cudaMalloc((void**)&d_G2, sizeof(dt)*m*ttRanks[2]);
        cudaStat2 = cudaHostAlloc((void**)&h_G2, sizeof(dt)*m*ttRanks[2],0);
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

    //1->2->3
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

    //1->2->3->4
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

    //1->2->3->4->5
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
        // printMatrix_Device(ttRanks[5], n, d_G6, ttRanks[5], "G6");
        if(d_A5     ) cudaFree(d_A5); d_A5 = NULL;
        cudaStat5 = cudaMemcpyAsync(h_G5, d_G5, sizeof(dt)*m*ttRanks[5], cudaMemcpyDeviceToHost,0);
        cudaStat6 = cudaMemcpyAsync(h_G6, d_G6, sizeof(dt)*n*ttRanks[5], cudaMemcpyDeviceToHost,0);
        // cudaStat6 = cudaDeviceSynchronize();
        assert(cudaStat5 == cudaSuccess);
        assert(cudaStat6 == cudaSuccess);
        if(d_G5     ) cudaFree(d_G5); d_G5 = NULL;
        if(d_G6     ) cudaFree(d_G6); d_G6 = NULL;

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
    printf("timeTransfer: %f\n", timeTransfer);
    float time = timer.seconds()/calculateTimes;
    return time;
}


int main(){
    for(int i = 40; i < 52; i=i+4){
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
        int ttRanks[7] = {2, mid_rank/2, mid_rank, mid_rank, mid_rank, mid_rank, 2};
        int ttDemns[6] = {k, k, k, k, k, k};

        int p = 8*8;
        if(k == 2)
            p=2;
        int m = k;
        long long n = pow(k,5);
        printf("%lld , mid_rank = %d, k = %d \n", sizeof(dt)*m*n, mid_rank,k);
        srand(1134);
        dt *h_A = NULL;
        cudaStat1 = cudaHostAlloc((void**)&h_A,sizeof(dt)*m*n,0);
        assert(cudaStat1 == cudaSuccess);
        generalTRTensor(h_A, ttRanks, ttDemns, p);
        // printMatrix(m, n, h_A, m, "hA1");
        warmupcu();
        if(k<40){
            ttdec_half_lu_device_6(h_A, ttRanks, ttDemns, 1, p, calError);
        }
        else{
            ttdec_half_lu_host_6(h_A, ttRanks, ttDemns, 1, p, calError);
        }
        // calculateTimes=0;
        float time = 0;
        if(k<40){
            time = ttdec_half_lu_device_6(h_A, ttRanks, ttDemns, calculateTimes, p, calError);
        }
        else{
            time = ttdec_half_lu_host_6(h_A, ttRanks, ttDemns, calculateTimes, p, calError);
        }

        printf("*************time****************\n %f \n*******************************\n", time);
     
        if(h_A      ) cudaFreeHost(h_A); h_A = NULL;  
    }
    return 0;
}
