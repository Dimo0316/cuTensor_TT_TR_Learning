#include "head.h"

__global__ void upper1(double *A,double *R,int m,int n)
{
  long long i = blockIdx.x*blockDim.x+threadIdx.x;
  const long long temp = blockDim.x*gridDim.x;
   while(i<n*n)
  { 
    long row=i/n;
    long col=i%n;
    if(row>=col)  
      R[i]=A[row*m+col];
    else
      R[i]=0;
    i+=temp;    
  }
  __syncthreads();
}
void basicQR(cusolverDnHandle_t cusolverH, double *d_A, double *d_R, int m, int n){
    // printf("start qr for m: %d, n:%d\n", m,n);
    double *d_work = NULL, *d_tau = NULL;
    int *devInfo = NULL;
    int  lwork = 0; 
    cudaMalloc((void**)&d_tau, sizeof(double)*n);
    cudaMalloc((void**)&devInfo, sizeof(int));
    cusolverDnDgeqrf_bufferSize(
        cusolverH, 
        m, 
        n, 
        d_A, 
        m, 
        &lwork);
    cudaMalloc((void**)&d_work, sizeof(double)*lwork );
    cusolverDnDgeqrf(
        cusolverH, 
        m, 
        n, 
        d_A, 
        m, 
        d_tau, 
        d_work, 
        lwork, 
        devInfo);
    cudaDeviceSynchronize();
    dim3 blockR((n*n+1024-1)/1024,1,1);
    upper1<<<blockR,1024>>>(d_A,d_R,m,n);

    cusolverDnDorgqr(
        cusolverH,
        m,
        n,
        n,
        d_A,
        m,
        d_tau,
        d_work,
        lwork,
        devInfo);
    cudaDeviceSynchronize();

    if (d_work) cudaFree(d_work); d_work = NULL;
    if (devInfo) cudaFree(devInfo); devInfo = NULL;
    if (d_tau) cudaFree(d_tau); d_tau = NULL;
}

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
/*forward_cuda(inlayer.value,inlayer.W,hidlayer_U2.value,hidlayer_U2.W,
                 hidlayer_B1.value,hidlayer_B1.W,hidlayer_U1.value,hidlayer_U1.W,outlayer.value,
                 k1,k2,
                 IPNNUM,HDNNUM,OPNNUM,BATCHSIZE,handle);*/
void forward_cuda(double *input,double *W_U2,double *outu2,double *W_B1,
                  double *outB1,double *W_U1,double *outU1,double *W2,double *outo,
                  double b1,double b2,
                  int in,int k2,int k1,int hid,int out,int batch,cublasHandle_t handle)
{
	//hid 行 in列 W1 ,out行，hid列 W2

	double alpha=1.0, beta=0.0;
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,k2,batch,in,&alpha,W_U2,k2,input,in,&beta,outu2,k2);
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,k1,batch,k2,&alpha,W_B1,k1,outu2,k2,&beta,outB1,k1);
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,hid,batch,k1,&alpha,W_U1,hid,outB1,k1,&beta,outU1,hid);


	//激活函数
	dim3 blockh((batch*hid+1024-1)/1024,1,1);
	activate<<<blockh,1024>>>(outU1,b1,hid*batch);
	cudaDeviceSynchronize();

	dim3 blocko((batch*out+1024-1)/1024,1,1);
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,out,batch,hid,&alpha,W2,out,outU1,hid,&beta,outo,out);	
	activate<<<blocko,1024>>>(outo,b2,out*batch);
	cudaDeviceSynchronize();
	//printTensor(outo,10,1,1);

}
/*
back_cuda(Tg,outlayer.value,
			hidlayer_U1.value,hidlayer_U1.W,
			hidlayer_B1.value,hidlayer_B1.W,
              hidlayer_U2.value,hidlayer_U2.W,
              inlayer.value,inlayer.W,IPNNUM,K2,K1,HDNNUM,OPNNUM,BATCHSIZE,handle
              );*/
void back_cuda(double *Y,double *Y_hat,
               double *outU1,double *W2,
               double *outB1,double *W_U1,
               double *outu2,double *W_B1,
               double *input,double *W_U2,int in,int k2,int k1,int hid,int out,int batch,double rate,cublasHandle_t handle)
{

	double *d_thta3,*d_thta2,*d_thta1,*d_thta0;
	cudaMalloc((void**)&d_thta3,sizeof(double)*out*batch);
	cudaMalloc((void**)&d_thta2,sizeof(double)*hid*batch);
	cudaMalloc((void**)&d_thta1,sizeof(double)*k1*batch);
	cudaMalloc((void**)&d_thta0,sizeof(double)*k2*batch);

	dim3 block2((batch*out+1024-1)/1024,1,1);
	Vector2_Multiply_By_Elements<<<block2,1024>>>(Y_hat, Y, out*batch, d_thta3);
	double alpha=1.0, beta=0.0;
	cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,hid,batch,out,&alpha,W2,out,d_thta3,out,&beta,d_thta2,hid);
	
	dim3 block1((batch*hid+1024-1)/1024,1,1);
	Vector1_Multiply_By_Elements<<<block1,1024>>>(outU1, d_thta2, hid*batch);
	cudaDeviceSynchronize();

	cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,k1,batch,hid,&alpha,W_U1,hid,d_thta2,hid,&beta,d_thta1,k1);
	cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,k2,batch,k1,&alpha,W_B1,k1,d_thta1,k1,&beta,d_thta0,k2);

	alpha=rate; beta=1.0;
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,out,hid,batch,&alpha,d_thta3,out,outU1,hid,&beta,W2,out);
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,hid,k1,batch,&alpha,d_thta2,hid,outB1,k1,&beta,W_U1,hid);	
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,k1,k2,batch,&alpha,d_thta1,k1,outu2,k2,&beta,W_B1,k1);
	cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,k2,in,batch,&alpha,d_thta0,k2,input,in,&beta,W_U2,k2);
	cudaDeviceSynchronize();

	cudaFree(d_thta0);
	cudaFree(d_thta1);
	cudaFree(d_thta2);
	cudaFree(d_thta3);
}	

double loss_gpu(double *A,double *B,int n,int batch,cublasHandle_t handle)
{	
	//A 实际值  B预测值

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

void basicSvd(cusolverDnHandle_t cusolverH, double *d_B, const int m, const int n, double *d_U, double *d_S, double *d_VT){
    assert(m>n);
    // double alpha = 1.0,beta = 0.0;	
    double *d_work = NULL, *d_rwork = NULL;
    int *devInfo = NULL;
    int lwork = 0;

    cudaMalloc((void**)&devInfo, sizeof(int));
	cusolverDnDgesvd_bufferSize(cusolverH,m,n,&lwork );
	cudaMalloc((void**)&d_work , sizeof(double)*lwork);

    cusolverDnDgesvd(cusolverH,'S','S',m,n,
        d_B,m,d_S,d_U,m,d_VT,n, // ldvt,
        d_work,lwork,d_rwork,devInfo);
    cudaDeviceSynchronize();
 
    if(d_work) cudaFree(d_work); d_work = NULL;
    if(devInfo) cudaFree(devInfo); devInfo = NULL;
    if(d_rwork) cudaFree(d_rwork); d_rwork = NULL;
}
void basicSvd_trans(cusolverDnHandle_t cusolverH, cublasHandle_t cublasH, double *d_B, const int m, const int n, double *d_UT, double *d_S, double *d_VT){
    double alpha = 1.0,beta = 0.0;  
    assert(m<n);
    // printf("start svd for m: %d, n:%d\n", m,n);
    double *d_BT = NULL, *d_U = NULL, *d_V=NULL;
    double *d_work = NULL, *d_rwork = NULL;
    int *devInfo = NULL;
    int lwork = 0;

    cudaMalloc((void**)&d_BT, sizeof(double)*m*n);
    cudaMalloc((void**)&d_U, sizeof(double)*m*m);
    cudaMalloc((void**)&d_V, sizeof(double)*m*n);
    cudaMalloc ((void**)&devInfo, sizeof(int));
   
//转置B
    cublasDgeam(cublasH,CUBLAS_OP_T, CUBLAS_OP_N, 
                n, m,&alpha,d_B, m,&beta,d_B, n,d_BT, n);
    cudaDeviceSynchronize();
    cusolverDnDgesvd_bufferSize(cusolverH,n,m,&lwork );
    cudaMalloc((void**)&d_work , sizeof(double)*lwork);

    signed char jobu = 'S'; // all m columns of U
    signed char jobvt = 'S'; // all n columns of VT
    // cout << "n: " << n <<" m:" << m <<endl;
    cusolverDnDgesvd(cusolverH,jobu,jobvt,n,m,
        d_BT,n,d_S,d_V,n,d_U,m, // ldvt,
        d_work,lwork,d_rwork,devInfo);
    cudaDeviceSynchronize();
    
//转置U 给UT
    cublasDgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, 
                            m, m,
                            &alpha,
                            d_U, m,
                            &beta,
                            d_U, m,
                            d_UT, m);
    cudaDeviceSynchronize();
    cublasDgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, 
                            m, n,
                            &alpha,
                            d_V, n,
                            &beta,
                            d_V, n,
                            d_VT, m);
    cudaDeviceSynchronize();

    if(d_BT) cudaFree(d_BT); d_BT = NULL;
    if(d_U) cudaFree(d_U); d_U = NULL;
    if(d_work) cudaFree(d_work); d_work = NULL;
    if(devInfo) cudaFree(devInfo); devInfo = NULL;
    if(d_rwork) cudaFree(d_rwork); d_rwork = NULL;
}

void tt(double *d_A,double *G1, double *G2,double *G3, int m, int n, int k1, int k2, cublasHandle_t handle)
{	
    cusolverDnHandle_t cusolverH = NULL;
    cusolverDnCreate(&cusolverH);
    cout << m << n <<endl;
    double *d_U1=NULL, *d_S1=NULL, *d_VT1=NULL, *d_A2=NULL;
    cudaMalloc((void**)&d_U1, sizeof(double)*m*m);
    cudaMalloc((void**)&d_S1, sizeof(double)*m);
    cudaMalloc((void**)&d_VT1, sizeof(double)*m*n);
    cudaMalloc((void**)&d_A2, sizeof(double)*k1*n);

    double *d_S2=NULL, *d_VT2=NULL;
    cudaMalloc((void**)&d_S2, sizeof(double)*k2);
    cudaMalloc((void**)&d_VT2, sizeof(double)*k2*n);

    basicSvd_trans(cusolverH, handle, d_A, m, n, d_U1, d_S1, d_VT1);

    cublasDcopy(handle, m*k1, d_U1, 1, G1, 1);
    cublasDdgmm(handle, CUBLAS_SIDE_LEFT, k1, n, d_VT1, m, d_S1, 1, d_A2, k1);
   	
    basicSvd_trans(cusolverH, handle, d_A2, k1, n, G2, d_S2, d_VT2);

    cublasDdgmm(handle, CUBLAS_SIDE_LEFT, k2, n, d_VT2, k2, d_S2, 1, G3, k2);

    cudaFree(d_U1);
    cudaFree(d_S1);
    cudaFree(d_VT1);
    cudaFree(d_A2);

    cudaFree(d_S2);
    cudaFree(d_VT2);
    // cudaFree(d_Gtemp);
    // cudaFree(d_Gtemp2);
    cusolverDnDestroy(cusolverH);
}

__global__ void Gsmaller(double *A,int m)
{
  long long i = blockIdx.x*blockDim.x+threadIdx.x;
  const long long temp = blockDim.x*gridDim.x;
   while(i<m)
  { 
    A[i] = A[i]*0.005;
    i+=temp;    
  }
  __syncthreads();
}

// void tt_random(double *d_A, double *G1, double *G2,double *G3, int m, int n, int k1, int k2, cublasHandle_t handle)
// {	
// 	cusolverDnHandle_t cusolverH = NULL;
//     cusolverDnCreate(&cusolverH);
//     double alpha = 1.0, beta = 0.0, alpha2 = 0.005;
// 	curandGenerator_t gen;
//     curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
//     curandSetPseudoRandomGeneratorSeed(gen, 1233456);
//     curandGenerateNormalDouble(gen, G1, n*k1, 0, 1); 
//     curandGenerateNormalDouble(gen, G2, k1*k2, 0, 1);
//     curandGenerateNormalDouble(gen, G3, m*k2, 0, 1);
//     // m = hid      n = input 
//     cout << m << n <<endl;
//     double *d_R1=NULL, *d_R2=NULL;
//     cudaMalloc((void**)&d_R1, sizeof(double)*n*k1);
//     cudaMalloc((void**)&d_R2, sizeof(double)*k1*k2);

//     basicQR(cusolverH, G1, d_R1, n, k1);
//     cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,k1,k1,k1,&alpha,d_R1,k1,G2,k1,&beta,G2,k1);

//     basicQR(cusolverH, G2, d_R2, k1, k1);
//     cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,k1,m,k1,&alpha2,d_R2,k1,G3,k1,&beta,G3,k1);

//     dim3 blockG1((n*k1+1024-1)/1024,1,1);
//     Gsmaller<<<blockG1,1024>>>(G1,n*k1);
//     dim3 blockG2((k1*k1+1024-1)/1024,1,1);
//     Gsmaller<<<blockG2,1024>>>(G2,k1*k1);

//     cudaFree(d_R1);
//     cudaFree(d_R2);
//     cusolverDnDestroy(cusolverH);
// }