#include "head.h"

__global__ void Vector2_Multiply_By_Elements (const double* a, const double* b, int n, double* out){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	while(tid<n)
	{
		out[tid]=a[tid]*(1.0-a[tid])*(b[tid]-a[tid]);
		tid+=temp;
	}
	__syncthreads();
}
__global__ void Vector1_Multiply_By_Elements (const double* a, double* b, int n){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	while(tid<n)
	{
		b[tid]=a[tid]*(1.0-a[tid])*b[tid];
		tid+=temp;
	}
	__syncthreads();
}

__global__ void activate(double *d_A,int b,int a)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	while(tid<a)
	{
		d_A[tid] = 1/(1+exp(-d_A[tid]+b));
		tid+=temp;
	}
	__syncthreads();
}

__global__ void dropout(double *d_A,double *d_B,int n)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	while(i<n)
	{
		d_A[i] = d_A[i] * d_B[i];
		i+=temp;
	}
	__syncthreads();
}

__global__ void relu(double *d_A,int b,int a)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	while(tid<a)
	{
		if(d_A[tid]<= 0)
			d_A[tid] = 0;
		tid+=temp;
	}
	__syncthreads();
}

void printTensor(double *d_des,long m,long n,long l){
	double *des = new double[m*n*l]();
	cudaMemcpy(des,d_des,sizeof(double)*m*n*l,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	for(long k = 0;k<l;k++){
		for(long i = 0;i<n;i++){
			for(long j = 0;j<m;j++){
				cout<<des[k*m*n+i*m+j]<<" ";
			}
			cout<<endl;
		}
		cout<<"~~~~~~~~~~~~~~~~"<<endl;
	}
	delete[] des;des=nullptr;
}
void warmup(){
	double *tmp = new double[9];
	for(unsigned i = 0; i < 9; ++i) {
		tmp[i] = i+1;
	}
	double *d_tmp;
	cudaMalloc((void**)&d_tmp,sizeof(double)*9);
	cudaMemcpy(d_tmp,tmp,sizeof(double)*9,cudaMemcpyHostToDevice);
	activate<<<1,512>>>(d_tmp,3,3);
	cudaDeviceSynchronize();
	cudaFree(d_tmp);

}
void forward_cuda(double *input,double *W1,double *outh,double *W2,double *outo,double k1,double k2,int in,int hid,int out,int batch,cublasHandle_t handle)
{
	//hid 行 in列 W1 ,out行，hid列 W2

	//printTensor(input,5,1,1);
	//cout<<"weigh matrix is :"<<endl;printTensor(W1,4,4,1);

	//double *outo;
	//cudaMalloc((void**)&outo,sizeof(double)*out);

	double alpha=1.0, beta=0.0;

	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,hid,batch,in,&alpha,W1,hid,input,in,&beta,outh,hid);

	//激活函数
	dim3 blockh((batch*hid+1024-1)/1024,1,1);
	activate<<<blockh,1024>>>(outh,k1,hid*batch);
	//relu<<<blockh,1024>>>(outh,k1,hid*batch);
	//printTensor(outh,3,3,1);
	cudaDeviceSynchronize();
/*
	std::default_random_engine e(time(0));
    std::bernoulli_distribution b(0.8);
    double *first = new double[hid*batch];
    for (int i = 0; i < hid*batch; ++i) {
        first[i] = (double)b(e);        
    }
    double *d;
    cudaMalloc((void**)&d,sizeof(double)*hid*batch);
    cudaMemcpy(d,first,sizeof(double)*hid*batch,cudaMemcpyHostToDevice);
    //printTensor(d,4,4,1);
    dropout<<<blockh,1024>>>(outh,d,hid*batch);
*/
	dim3 blocko((batch*out+1024-1)/1024,1,1);
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,out,batch,hid,&alpha,W2,out,outh,hid,&beta,outo,out);	
	activate<<<blocko,1024>>>(outo,k2,out*batch);

	//printTensor(outo,4,4,1);
	cudaDeviceSynchronize();

}

void back_cuda(double *Y,double *Y_hat,double *outh,double *W2,double *input,double *W1,int in,int hid,int out,int batch,double rate,cublasHandle_t handle)
{
	double *d_thta3,*d_thta2;
	cudaMalloc((void**)&d_thta3,sizeof(double)*out*batch);
	cudaMalloc((void**)&d_thta2,sizeof(double)*hid*batch);
	//printTensor(Y,10,1,1);
	dim3 block2((batch*out+1024-1)/1024,1,1);
	Vector2_Multiply_By_Elements<<<block2,1024>>>(Y_hat, Y, out*batch, d_thta3);
	double alpha=1.0, beta=0.0;
	cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,hid,batch,out,&alpha,W2,out,d_thta3,out,&beta,d_thta2,hid);
	dim3 block1((batch*hid+1024-1)/1024,1,1);
	Vector1_Multiply_By_Elements<<<block1,1024>>>(outh, d_thta2, hid*batch);
	cudaDeviceSynchronize();
	alpha=rate; beta=1.0;
	//cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,hid,in,1,&alpha,d_thta2,hid,input,1,&beta,W1,hid);
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,hid,in,batch,&alpha,d_thta2,hid,input,in,&beta,W1,hid);

	//cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,out,hid,1,&alpha,d_thta3,out,outh,1,&beta,W2,out);
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,out,hid,batch,&alpha,d_thta3,out,outh,hid,&beta,W2,out);
	//cout<<"model bp1:"<<endl;printTensor(W1,4,4,1);
	//cout<<"model bp2:"<<endl;printTensor(W2,4,4,1);
	cudaFree(d_thta2);
	cudaFree(d_thta3);
}	
double loss_gpu(double *A,double *B,int n,int batch,cublasHandle_t handle)
{	
	//A 实际值  B预测值
	//printTensor(A,2,2,1);
	//printTensor(B,2,2,1);
	double alpha1 = -1.0,loss;
	double *tmp;
	cudaMalloc((void**)&tmp,sizeof(double)*n*batch);

	cublasDcopy(handle,n*batch,B,1,tmp,1);

	cublasDaxpy(handle,n*batch,&alpha1,A,1,tmp,1);
	cublasDnrm2(handle,n*batch,tmp,1,&loss); 
	cudaDeviceSynchronize();
	cudaFree(tmp);
	return loss;
}

