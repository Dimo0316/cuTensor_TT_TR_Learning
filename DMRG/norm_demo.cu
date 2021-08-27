#include<iostream>
#include<fstream>
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<cusolverDn.h>
#include<curand.h>
#include<cmath>
using namespace std;
typedef float dt;
__global__ void initIdeMat(float *AA,int m){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  const int temp = blockDim.x*gridDim.x;
  while(i<m*m){
    int row = i%m;
    int col = i/m;
    if(row==col){
      AA[col*m+row] = 1;
    }else{
      AA[col*m+row] = 0;
    }
    i+=temp;
  }
  __syncthreads();
}
__global__ void Init_P(float *d_P,int m)
{
  long long i = blockIdx.x*blockDim.x+threadIdx.x;
  const long long temp = blockDim.x*gridDim.x;

  while(i<m*m)
  {
    int row = i%m;
    int col = i/m;
    if(row+col+1 == m){
      d_P[i] = 1;
    }else{
      d_P[i] = 0;
    }
    i+=temp;
    
  }
  __syncthreads();
}

__global__ void tensorToMode3(dt *T1,dt *T2,int m,int n,int k){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  const int temp = blockDim.x*gridDim.x;
  while(i<m*n*k){
    int tube = i/(m*n);
    int row = (i-tube*(m*n))%m;
    int col = (i-tube*(m*n))/m;
    T2[k*(col*m+row)+tube] = T1[tube*m*n+col*m+row];
    i+=temp;
  }
    __syncthreads();
}

int main()
{
 dt *A = new dt[2*2*2];

 dt *d_A,*d_A2;
 cudaMalloc((void**)&d_A,sizeof(dt)*8);
 cudaMalloc((void**)&d_A2,sizeof(dt)*8);

 for(int i=0;i<8;i++)
 {
  A[i] = i+1;
 }

 cudaMemcpy(d_A,A,sizeof(dt)*8,cudaMemcpyHostToDevice);
 tensorToMode3<<<50,50>>>(d_A,d_A2,2,2,2);

 cudaMemcpy(A,d_A2,sizeof(dt)*8,cudaMemcpyDeviceToHost);

 for(int i=0;i<8;i++)
 {
  cout<<A[i]<<" _ ";
 }

 




}