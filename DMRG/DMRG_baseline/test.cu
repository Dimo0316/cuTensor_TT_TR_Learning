#include<iostream>
#include<fstream>
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include<cusolverDn.h>
#include<curand.h>
#include<cmath>
using namespace std;
typedef float dt;
__global__ void tensorToMode1(dt *T1,dt *T2,int m,int n,int k ){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  const int temp = blockDim.x*gridDim.x;
  while(i<m*n*k){
    int tube = i/(m*n);
    int row = (i-tube*(m*n))%m;
    int col = (i-tube*(m*n))/m;
    T2[tube*m*n+col*m+row] = T1[tube*m*n+col*m+row];
    i+=temp;
  }
  __syncthreads();  
}
__global__ void transpose(float *A,float* AT,int m,int n){
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  const int temp = blockDim.x*gridDim.x;
  while(i<n*m){
    int row = i/n;
    int col = i%n;
    AT[col*m+row] = A[row*n+col];
    i+=temp;
  }
    __syncthreads();
}

int main() {

	dt *A = new dt[3*3*3];
	dt *B = new dt[3*3*3];
	for(int i=0;i<27;i++)
	{
		A[i]=i+1;
	}
	dt *d_A,*d_A_mode,*d_A_modeT;
	cudaMalloc((void**)&d_A,sizeof(dt)*27);
	cudaMalloc((void**)&d_A_mode,sizeof(dt)*27);
	cudaMalloc((void**)&d_A_modeT,sizeof(dt)*27);
	cudaMemcpy(d_A,A,sizeof(dt)*27,cudaMemcpyHostToDevice);
	tensorToMode1<<<50,50>>>(d_A,d_A_mode,3,3,3);
	transpose<<<50,50>>>(d_A_mode,d_A_modeT,3*3,3);
	cudaDeviceSynchronize();

	cudaMemcpy(B,d_A_modeT,sizeof(dt)*27,cudaMemcpyDeviceToHost);

	for(int i = 0; i < 27; ++i) {
		/* code */
		cout<<B[i]<<"_";
	}
	cout<<endl;

	cudaFree(d_A_mode);
	cudaFree(d_A_modeT);	
	return 0;
}