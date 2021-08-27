#include<iostream>
#include<fstream>
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<cusolverDn.h>
#include<curand.h>
#include<cmath>
using namespace std;
typedef float dt;

__global__  void floattohalf(dt *AA,half *BB,long m){
	long i = blockIdx.x*blockDim.x+threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	if(i<m){
		BB[i]=__float2half(AA[i]);
		i+=temp;
	}
	__syncthreads();
}

void f2h(dt *A,half *B,long num){
	dim3 threads(512,1,1);
	dim3 blocks((num+512-1)/512,1,1);	
	floattohalf<<<blocks,threads>>>(A,B,num);
}

int main()
{
	for(int hh = 100;hh<160;hh=hh+160)
	{
	dt *A,*B,*C;
	int a=hh;
	int r=hh;
	int b=hh;
	cudaHostAlloc((void**)&A,sizeof(dt)*a*r,0);
	cudaHostAlloc((void**)&B,sizeof(dt)*r*b,0);
	cudaHostAlloc((void**)&C,sizeof(dt)*b*a,0);
	srand(5);
	for(long long i = 0;i<a*r;i++){
		A[i] = rand()*0.1/(RAND_MAX*0.1);		
	}	
	for(long long i = 0;i<r*b;i++){
		B[i] = rand()*0.1/(RAND_MAX*0.1);		
	}

	dt alpha = 1.0;
	dt beta = 0.0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	//cublasMath_t CUBLAS_TENSOR_OP_MATH;
	cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH);

	dt *d_A;
	dt *d_B;
	dt *d_C;

		
		half *h_A,*h_B;
		cudaMalloc((void **)&h_A,sizeof(half)*a*r);
		cudaMalloc((void **)&h_B,sizeof(half)*b*r);

	cudaMalloc((void **)&d_A,sizeof(dt)*a*r);
	cudaMalloc((void **)&d_B,sizeof(dt)*r*b);
	cudaMalloc((void **)&d_C,sizeof(dt)*a*b);
	cudaMemcpyAsync(d_A,A,sizeof(dt)*a*r,cudaMemcpyHostToDevice,0);
	cudaMemcpyAsync(d_B,B,sizeof(dt)*r*b,cudaMemcpyHostToDevice,0);
	cudaDeviceSynchronize();

		f2h(d_A,h_A,a*r);
		f2h(d_B,h_B,b*r);
		cudaDeviceSynchronize();

	cublasGemmEx(handle,CUBLAS_OP_N,
		             CUBLAS_OP_N,
		             a,
		             b,
		             r,
		             &alpha,
		             h_A,
		             CUDA_R_16F,
		             a,
		             h_B,
		             CUDA_R_16F,
		             r,
		             &beta,
		             d_C,
		             CUDA_R_32F,
		             a,
		             CUDA_R_32F,
		             CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	cudaDeviceSynchronize();

	cudaMemcpy(C,d_C,sizeof(dt)*a*b,cudaMemcpyDeviceToHost);

	for(int i=0;i<10;i++)
	{
		cout<<C[i]<<" _ ";
	}
	cout<<endl;
}
}

