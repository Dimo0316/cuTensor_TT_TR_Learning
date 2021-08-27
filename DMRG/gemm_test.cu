#include<iostream>
#include<fstream>
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<cusolverDn.h>
#include<curand.h>
#include<cmath>
#include <cutensor.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>
#include<ctime>

using namespace std;
typedef float dt;


void permute(dt *d_L,int a,int b,int c,int d,int e,cutensorHandle_t tensor_handle,int flag)
{	
	cudaDataType_t typeA = CUDA_R_32F;
    cudaDataType_t typeC = CUDA_R_32F;
    cudaDataType_t typeCompute = CUDA_R_32F;

    std::vector<int> modeA{'a','b','c','d','e'};
    std::vector<int> modeC(5);
    if(flag == 0)
    {    	
    	//std::vector<int> modeC{'e','a','d','b','c'};
    	modeC[0]='e';modeC[1]='a';modeC[2]='d';modeC[3]='b';modeC[4]='c';
    }
    else
    {   	
    	//std::vector<int> modeC{'a','d','e','b','c'};
    	modeC[0]='a';modeC[1]='d';modeC[2]='e';modeC[3]='b';modeC[4]='c';
    }

    int nmodeA = modeA.size();
    int nmodeC = modeC.size();
    std::unordered_map<int, int64_t> extent;

    extent['a'] = a;
    extent['b'] = b;
    extent['c'] = c;
    extent['d'] = d;
    extent['e'] = e;

    std::vector<int64_t> extentA;
    for (auto mode : modeA)
        extentA.push_back(extent[mode]);
    std::vector<int64_t> extentC;
    for (auto mode : modeC)
        extentC.push_back(extent[mode]);

    size_t elementsA = 1;
    for (auto mode : modeA)
        elementsA *= extent[mode];
    size_t elementsC = 1;
    for (auto mode : modeC)
        elementsC *= extent[mode];

    //size_t sizeA = sizeof(float) * elementsA;
    size_t sizeC = sizeof(float) * elementsC;

    dt *d_C;
    cudaMalloc((void**)&d_C,sizeC);

    cutensorHandle_t handle;
    cutensorInit(&handle);

    cutensorTensorDescriptor_t descA;
    cutensorInitTensorDescriptor(&handle,
                 &descA,
                 nmodeA,
                 extentA.data(),
                 NULL /* stride */,
                 typeA, CUTENSOR_OP_IDENTITY);
    cutensorTensorDescriptor_t descC;
    cutensorInitTensorDescriptor(&handle,
                 &descC,
                 nmodeC,
                 extentC.data(),
                 NULL /* stride */,
                 typeC, CUTENSOR_OP_IDENTITY);
    const float one = 1.0f;
    cutensorPermutation(&handle,
                 &one, d_L, &descA, modeA.data(),
                       d_C, &descC, modeC.data(),
                 typeCompute, 0 /* stream */);
    cudaDeviceSynchronize();

    d_L = d_C;
    if (d_C) cudaFree(d_C);
}
void ncon(dt *d_L,dt *d_M,dt *d_A,int n,dt *d_out,cublasHandle_t handle,cutensorHandle_t tensor_handle)
{	
	dt *d_LM;
	cudaMalloc((void**)&d_LM,sizeof(dt)*n *n*n*n*n);

	dt alpha = 1.0;
	dt beta = 0.0;
	cublasSgemm(handle,
	            CUBLAS_OP_N,CUBLAS_OP_T,
	            n* n,n*n*n,n,
	            &alpha,d_L,n*n,d_M,n*n*n,
	            &beta,d_LM,n* n
	            );
	cudaDeviceSynchronize();

	//[2,4,-2][1,2,4,5,-3]  
	// d_LM permute ->[-3,1,5,2,4]
	// (A+)(LM)
	permute(d_LM,n,n,n,n,n,tensor_handle,0);
	dt *d_ALM;
	cudaMalloc((void**)&d_ALM,sizeof(dt)*n*n*n*n);
	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,
	            n,n*n*n,n*n,
	            &alpha,d_A,n*n,d_LM,n*n*n,
	            &beta,d_ALM,n
	            );
	// A * ALM 
	//[1,5,-1][-2,-3,1,5]
  	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,
  	            n,n*n,n*n,
  	            &alpha,d_A,n*n,d_ALM,n*n,
  	            &beta,d_out,n
  	            );
  	
  	cudaDeviceSynchronize();

  	cudaFree(d_LM);
  	cudaFree(d_ALM);
}
int main()
{
	int n = 5;

    float *A = new float[n*n*n];
    float *B = new float[n*n*n*n];
    float *C = new float[n*n*n];
    float *D = new float[n*n*n]; // C=D

    for (int i = 0; i < n*n*n; i++)
        A[i] = (((float) rand())/RAND_MAX - 0.5)*100;
    for (int i = 0; i < n*n*n*n; i++)
        B[i] = (((float) rand())/RAND_MAX - 0.5)*100;
    for (int i = 0; i < n*n*n; i++)
        C[i] = (((float) rand())/RAND_MAX - 0.5)*100;
    for (int i = 0; i < n*n*n; i++)
       D[i]=C[i];

    float *A_d,*B_d,*C_d;
    cudaMalloc((void**)&A_d,sizeof(float)*n*n*n);
    cudaMalloc((void**)&B_d,sizeof(float)*n*n*n*n);
    cudaMalloc((void**)&C_d,sizeof(float)*n*n*n);
    
    cudaMemcpy(A_d,A,sizeof(float)*n*n*n,cudaMemcpyHostToDevice);
    cudaMemcpy(B_d,B,sizeof(float)*n*n*n,cudaMemcpyHostToDevice);
    cudaMemcpy(C_d,C,sizeof(float)*n*n*n,cudaMemcpyHostToDevice);

    float *out_d;
    cudaMalloc((void**)&out_d,sizeof(float)*n*n*n);

	cublasHandle_t handle;
	cublasCreate(&handle);
	cutensorHandle_t tensor_handle;
    cutensorInit(&tensor_handle); 

     float time_elapsed=0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord( start,0);

    ncon(A_d,B_d,C_d,n,out_d,handle,tensor_handle);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);

    cout<<"cost time is :"<<time_elapsed<<endl;





}