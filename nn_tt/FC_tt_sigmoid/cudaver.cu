// #pragma once
#include "head.h"

__global__ void Vector2_Multiply_By_Elements (const dt* a, const dt* b, int n, dt* out){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	while(tid<n)
	{
		out[tid]=a[tid]*(1.0-a[tid])*(a[tid]-b[tid]);
		tid+=temp;
	}
	__syncthreads();
}

__global__ void Vector1_Multiply_By_Elements (const dt* a, dt* b, int n){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	while(tid<n)
	{	
		b[tid]=a[tid]*(1.0-a[tid])*b[tid];
		tid+=temp;
	}
	__syncthreads();
}

__global__ void activate_sigmoid(dt *d_A,int b,int a)
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

__global__ void activate_relu(dt *d_A,int a)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	while(tid<a)
	{
		// d_A[tid] = 1/(1+exp(-d_A[tid]));
		if(d_A[tid] < FLOATZERO)
			d_A[tid] = 0;
		tid+=temp;
	}
	__syncthreads();
}

__global__ void activate_softmax_exp(dt *d_A,int a)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	while(tid<a)
	{
		d_A[tid] = exp(d_A[tid]);
		tid+=temp;
	}
	__syncthreads();
}

__global__ void activate_softmax_sum(dt *d_sum, dt *d_A, int batch){
	// __shared__ dt sharedMem[];
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	while(tid < batch)
	{	
		d_sum[tid] = d_A[tid*10]+d_A[tid*10+1]+d_A[tid*10+2]+d_A[tid*10+3]+d_A[tid*10+4]+d_A[tid*10+5]+d_A[tid*10+6]+d_A[tid*10+7]+d_A[tid*10+8]+d_A[tid*10+9];
		tid+=temp;
	}
	__syncthreads();
}

__global__ void activate_softmax_final(dt *d_A, dt *d_Asum, int a)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	while(tid<a)
	{
		d_A[tid] = d_A[tid]/d_Asum[tid/10];
		tid+=temp;
	}
	__syncthreads();
}

void activate_softmax(dt *d_A, int out, int batch){
	dt* d_Asum10 = NULL;
	printf("\nd_A\n");
	printTensor(d_A,10,1,1);
	cudaMalloc((void**)&d_Asum10, sizeof(dt)*batch);
	dim3 blocko((batch*out+1024-1)/1024,1,1);
	activate_softmax_exp<<<blocko,1024>>>(d_A, out*batch);
	printf("\nd_A\n");
	printTensor(d_A,10,1,1);

	dim3 blocksum((batch+256-1)/256,1,1);
	activate_softmax_sum<<<blocksum, 256>>>(d_Asum10, d_A, batch);
	cudaDeviceSynchronize();

	printf("\nd_Asum10\n");
	printTensor(d_Asum10,10,1,1);
	activate_softmax_final<<<blocko,1024>>>(d_A, d_Asum10, out*batch);
	cudaDeviceSynchronize();
	printf("\nd_A\n");
	printTensor(d_A,10,1,1);
	if(d_Asum10) cudaFree(d_Asum10); d_Asum10 = NULL;
}

void printTensor(dt *d_des,long m,long n,long l){
	dt *des = new dt[m*n*l]();
	cudaMemcpy(des,d_des,sizeof(dt)*m*n*l,cudaMemcpyDeviceToHost);
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
	dt *tmp = new dt[9];
	for(unsigned i = 0; i < 9; ++i) {
		tmp[i] = i+1;
	}
	dt *d_tmp;
	cudaMalloc((void**)&d_tmp,sizeof(dt)*9);
	cudaMemcpy(d_tmp,tmp,sizeof(dt)*9,cudaMemcpyHostToDevice);
	activate_sigmoid<<<1,512>>>(d_tmp,3,3);
	cudaDeviceSynchronize();
	cudaFree(d_tmp);
}

__global__ void loss_gpu_cross_kernel(dt *d_temp, dt *traget, dt *out, int number){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const long temp = blockDim.x*gridDim.x;
	while(tid<number)
	{	
		if(traget[tid] == 0.0)
			d_temp[tid] = 0.0;
		else
			d_temp[tid] = traget[tid]*log(out[tid]);
		tid+=temp;
	}
	__syncthreads();
}


dt loss_gpu_cross(dt *A, dt *B, int n, int batch, cublasHandle_t handle)
{	
	//A 实际值  B预测值
	printf("\nT\n");
	printTensor(A,10,1,1);
	printf("\nO\n");
	printTensor(B,10,1,1);
	dt alpha1 = -1.0,loss;
	dt *tmp;
	cudaMalloc((void**)&tmp,sizeof(dt)*n*batch);

	cublasScopy(handle,n*batch,B,1,tmp,1);

	cublasSaxpy(handle,n*batch,&alpha1,A,1,tmp,1);
	cublasSnrm2(handle,n*batch,tmp,1,&loss); 
	cudaDeviceSynchronize();
	cudaFree(tmp);
	return loss;
}

void forward_cuda(dt *input, dt *G1, dt *outG1, dt *G2,
                  dt *outG2, dt *G3, dt *outG3, dt *G4,
                  dt *outG4, dt *W2, dt* outFinal, dt b1,dt b2,
                  int* inp_modes, int* out_modes, int* ranks, int* out_values, int out, int batch, cublasHandle_t handle)
{	
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	dt *outG1_mid = NULL, *outG2_mid = NULL, *outG3_mid = NULL, *outG4_mid = NULL;
	cudaMalloc((void**)&outG1_mid, sizeof(dt)*out_modes[0]*ranks[1]*out_values[0]*batch);
	cudaMalloc((void**)&outG2_mid, sizeof(dt)*out_modes[1]*ranks[2]*out_values[1]*batch);
	cudaMalloc((void**)&outG3_mid, sizeof(dt)*out_modes[2]*ranks[3]*out_values[2]*batch);
	cudaMalloc((void**)&outG4_mid, sizeof(dt)*out_modes[3]*ranks[4]*out_values[3]*batch);

	//hid 行 in列 W1 ,out行，hid列 W2
	dt alpha=1.0, beta=0.0;
	cublas_status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
				out_modes[0]*ranks[1], out_values[0]*batch, ranks[0]*inp_modes[0], 
				&alpha, G1, out_modes[0]*ranks[1], input, ranks[0]*inp_modes[0], &beta, outG1_mid, out_modes[0]*ranks[1]); //outG1 = G1*input
	assert(cublas_status == CUBLAS_STATUS_SUCCESS);
	cublas_status = cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, ranks[1]*out_values[0]*batch, out_modes[0],
                &alpha, outG1_mid, out_modes[0], &beta, outG1, ranks[1]*out_values[0]*batch, outG1, ranks[1]*out_values[0]*batch);
	assert(cublas_status == CUBLAS_STATUS_SUCCESS);
	printf("outG1\n");
	printTensor(outG1,10,1,1);

	cublas_status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
				out_modes[1]*ranks[2], out_values[1]*batch, ranks[1]*inp_modes[1], 
				&alpha, G2, out_modes[1]*ranks[2], outG1, ranks[1]*inp_modes[1], &beta, outG2_mid, out_modes[1]*ranks[2]); //outG2 = G2*outG1
	assert(cublas_status == CUBLAS_STATUS_SUCCESS);
	cublas_status = cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, ranks[2]*out_values[1]*batch, out_modes[1],
                &alpha, outG2_mid, out_modes[1], &beta, outG2, ranks[2]*out_values[1]*batch, outG2, ranks[2]*out_values[1]*batch);
	assert(cublas_status == CUBLAS_STATUS_SUCCESS);
	printf("outG2\n");
	printTensor(outG2,10,1,1);

	cublas_status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
				out_modes[2]*ranks[3], out_values[2]*batch, ranks[2]*inp_modes[2], 
				&alpha, G3, out_modes[2]*ranks[3], outG2, ranks[2]*inp_modes[2], &beta, outG3_mid, out_modes[2]*ranks[3]); //outG3 = G3*outG2
	assert(cublas_status == CUBLAS_STATUS_SUCCESS);
	cublas_status = cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, ranks[3]*out_values[2]*batch, out_modes[2],
                &alpha, outG3_mid, out_modes[2], &beta, outG3, ranks[3]*out_values[2]*batch, outG3, ranks[3]*out_values[2]*batch);
	assert(cublas_status == CUBLAS_STATUS_SUCCESS);
	printf("outG3\n");
	printTensor(outG3,10,1,1);

	cublas_status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
				out_modes[3]*ranks[4], out_values[3]*batch, ranks[3]*inp_modes[3], 
				&alpha, G4, out_modes[3]*ranks[4], outG3, ranks[3]*inp_modes[3], &beta, outG4_mid, out_modes[3]*ranks[4]); //outG4 = G4*outG3
	assert(cublas_status == CUBLAS_STATUS_SUCCESS);
	cublas_status = cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, ranks[4]*out_values[3]*batch, out_modes[3],
                &alpha, outG4_mid, out_modes[3], &beta, outG4, ranks[4]*out_values[3]*batch, outG4, ranks[4]*out_values[3]*batch);
	assert(cublas_status == CUBLAS_STATUS_SUCCESS);
	printf("outG4\n");
	printTensor(outG4,10,1,1);

	int hid = out_modes[0] * out_modes[1] * out_modes[2] * out_modes[3];
	//激活函数
	dim3 blockh((batch*hid+1024-1)/1024,1,1);
	activate_sigmoid<<<blockh,1024>>>(outG4, b1, hid*batch);
	cudaDeviceSynchronize();
	printf("relu outG4\n");
	printTensor(outG4,10,1,1);
	dim3 blocko((batch*out+1024-1)/1024,1,1);
	cublas_status = cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,out,batch,hid,&alpha,W2,out,outG4,hid,&beta,outFinal,out);	
	assert(cublas_status == CUBLAS_STATUS_SUCCESS);

	activate_sigmoid<<<blocko,1024>>>(outFinal, b2, out*batch);
	// cudaDeviceSynchronize();

	// printTensor(outFinal,10,1,1);
	if(outG1_mid) cudaFree(outG1_mid); outG1_mid=NULL;
	if(outG2_mid) cudaFree(outG2_mid); outG2_mid=NULL;
	if(outG3_mid) cudaFree(outG3_mid); outG3_mid=NULL;
	if(outG4_mid) cudaFree(outG4_mid); outG4_mid=NULL;
}


void back_cuda(dt *Y,dt *Y_hat, dt *W2,
               dt *outG4, dt *G4, dt *outG3, dt *G3,
               dt *outG2, dt *G2, dt *outG1, dt *G1,
               dt *input, 
               int* inp_modes, int* out_modes, int* ranks, int* out_values, int out, int batch, dt rate, cublasHandle_t handle){
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	int hid = out_modes[0] * out_modes[1] * out_modes[2] * out_modes[3];
	dt *d_thta1,*d_thta2,*d_thta3,*d_thta4,*d_thta5;
	cudaMalloc((void**)&d_thta5,sizeof(dt)*out*batch);
	cudaMalloc((void**)&d_thta4,sizeof(dt)*hid*batch);

	cudaMalloc((void**)&d_thta3,sizeof(dt)*ranks[3]*inp_modes[3]*out_values[3]*batch);
	cudaMalloc((void**)&d_thta2,sizeof(dt)*ranks[2]*inp_modes[2]*out_values[2]*batch);
	cudaMalloc((void**)&d_thta1,sizeof(dt)*ranks[1]*inp_modes[1]*out_values[1]*batch);


	dim3 block2((batch*out+1024-1)/1024,1,1);
	Vector2_Multiply_By_Elements<<<block2,1024>>>(Y_hat, Y, out*batch, d_thta5);
	dt alpha=1.0, beta=0.0;
	cublas_status = cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,hid,batch,out,&alpha,W2,out,d_thta5,out,&beta,d_thta4,hid);
	
	dim3 block1((batch*hid+1024-1)/1024,1,1);
	Vector1_Multiply_By_Elements<<<block1,1024>>>(outG4, d_thta4, hid*batch);
	cudaDeviceSynchronize();

///TODO /修改
	cublas_status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
				ranks[3]*inp_modes[3], out_values[3]*batch, out_modes[3]*ranks[4],
				&alpha, G4, out_modes[3]*ranks[4], d_thta4, out_modes[3]*ranks[4], &beta, d_thta3, ranks[3]*inp_modes[3]);
	assert(cublas_status == CUBLAS_STATUS_SUCCESS);
	cublas_status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
				ranks[2]*inp_modes[2], out_values[2]*batch, out_modes[2]*ranks[3],
				&alpha, G3, out_modes[2]*ranks[3], d_thta3, out_modes[2]*ranks[3], &beta, d_thta2, ranks[2]*inp_modes[2]);
	assert(cublas_status == CUBLAS_STATUS_SUCCESS);
	cublas_status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
				ranks[1]*inp_modes[1], out_values[1]*batch, out_modes[1]*ranks[2],
				&alpha, G2, out_modes[1]*ranks[2], d_thta2, out_modes[1]*ranks[2], &beta, d_thta1, ranks[1]*inp_modes[1]);
	assert(cublas_status == CUBLAS_STATUS_SUCCESS);


	alpha=rate; beta=1.0;
	cublas_status = cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,out,hid,batch,&alpha,d_thta1,out,outG4,hid,&beta,W2,out);
	assert(cublas_status == CUBLAS_STATUS_SUCCESS);

///TODO /修改
	cublas_status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
				out_modes[3]*ranks[4], ranks[3]*inp_modes[3], out_values[3]*batch,
				&alpha, d_thta4, out_modes[3]*ranks[4], outG3, ranks[3]*inp_modes[3], &beta, G4, out_modes[3]*ranks[4]);
	assert(cublas_status == CUBLAS_STATUS_SUCCESS);
	cublas_status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
				out_modes[2]*ranks[3], ranks[2]*inp_modes[2], out_values[2]*batch,
				&alpha, d_thta3, out_modes[2]*ranks[3], outG2, ranks[2]*inp_modes[2], &beta, G3, out_modes[2]*ranks[3]);
	assert(cublas_status == CUBLAS_STATUS_SUCCESS);
	cublas_status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
				out_modes[1]*ranks[2], ranks[1]*inp_modes[1], out_values[1]*batch,
				&alpha, d_thta2, out_modes[1]*ranks[2], outG1, ranks[1]*inp_modes[1], &beta, G2, out_modes[1]*ranks[2]);
	assert(cublas_status == CUBLAS_STATUS_SUCCESS);
	cublas_status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
				out_modes[0]*ranks[1], ranks[0]*inp_modes[0], out_values[0]*batch,
				&alpha, d_thta1, out_modes[0]*ranks[1], input, ranks[0]*inp_modes[0], &beta, G1, out_modes[0]*ranks[1]);
	assert(cublas_status == CUBLAS_STATUS_SUCCESS);
	cudaDeviceSynchronize();


	cudaFree(d_thta1);
	cudaFree(d_thta2);
	cudaFree(d_thta3);
	cudaFree(d_thta4);
	cudaFree(d_thta5);
}	
