#include "head.h"

void QR(dt *d_A,dt *d_R,int m,int n,cusolverDnHandle_t cusolverH)
{

	dim3 threads(1024,1,1);
	dim3 block0((m*n+1024-1)/1024,1,1);

	dt *d_tau = NULL;
	int *devInfo = NULL;
	dt *d_work = NULL;
	int lwork_geqrf = 0;
	int lwork_orgqr = 0;
	int lwork = 0;

	dt *d_AT;
	cudaMalloc((void**)&d_AT,sizeof(dt)*m*n);
	transpose<<<block0,threads>>>(d_A,d_AT,m,n);
	cudaDeviceSynchronize();


	cudaMalloc ((void**)&d_tau, sizeof(dt)*n);
	cudaMalloc ((void**)&devInfo, sizeof(int));

	cusolverDnSgeqrf_bufferSize(cusolverH,
        m,n,
        d_A,m,&lwork_geqrf);
	cusolverDnSorgqr_bufferSize(cusolverH,
        m,n,n,
        d_A,m,d_tau,
        &lwork_orgqr);
	lwork = (lwork_geqrf > lwork_orgqr)? lwork_geqrf : lwork_orgqr;
	cudaMalloc((void**)&d_work, sizeof(dt)*lwork);
	cusolverDnSgeqrf(cusolverH,
        m,n,
        d_A,m,d_tau,
        d_work,lwork,devInfo);
	cudaDeviceSynchronize();
	//R 存储在 A的上三角阵（包括主对角线）
	upper<<<block0,threads>>>(d_A,d_R,m,n);
	cudaDeviceSynchronize();

	cusolverDnSorgqr(cusolverH,
	    m,n,n,
        d_A,m,
        d_tau,d_work,
        lwork,devInfo);
	cudaDeviceSynchronize();

	cudaFree(d_tau);
	cudaFree(d_work);
	cudaFree(devInfo);
	cudaFree(d_AT);	
	//cout<<"QR finish~~~~~~~~~~~~~"<<endl;
}

void RQ(dt *d_A,dt *d_Q,dt *d_QRweight,int a,int b,int c,cublasHandle_t handle,cusolverDnHandle_t cusolverH)
{

	dim3 threads(1024,1,1);
	dim3 block0((a*a+1024-1)/1024,1,1);


	dt *p_Q = new dt[a*b*c];
	dt *p_R = new dt[a*a];
	dt alpha = 1.0;
	dt beta = 0.0;
	// d_A  a*(b*c)
	//初始一个用于交换行的矩阵p  a*a 
	dt *d_P; // (b*c)*a   Q-> (b*c)*a   R-> a*a
	cudaMalloc((void**)&d_P,sizeof(dt)*a*a);
	Init_P<<<block0,threads>>>(d_P,a);
	// (P * A)T
	dt *d_PAT;
	cudaMalloc((void**)&d_PAT,sizeof(dt)*a*b*c);
	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,
	            b*c,a,a,
	            &alpha,d_A,a,d_P,a,
	            &beta,d_PAT,b*c
	            );
	// 对 d_PAT 进行QR分解，得到 Q,R

	QR(d_PAT,d_QRweight,b*c,a,cusolverH);
   //  Q = p * QT
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,
	            a,b*c,a,
	            &alpha,d_P,a,d_PAT,b*c,
	            &beta,d_Q,a
	            );
	cudaDeviceSynchronize();

	//R = p * RT * P
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,
	            a,a,a,
	            &alpha,d_P,a,d_QRweight,a,
	            &beta,d_QRweight,a
	            );
	cudaDeviceSynchronize();
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
	            a,a,a,
	            &alpha,d_QRweight,a,d_P,a,
	            &beta,d_QRweight,a
	            );
	cudaDeviceSynchronize();

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
__global__ void upper(float *A,float *R,int m,int n)
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

float norm(dt *d_l,dt sh,int len,cublasHandle_t handle)
{
	cublasSnrm2(handle,len,d_l,1,&sh);
	cudaDeviceSynchronize();

	return sh;
}

void dgmm(dt *d_S,dt *d_VH,dt *d_svh,int m,int n,cublasHandle_t handle)
{	
	 cublasSdgmm(handle,
	             CUBLAS_SIDE_LEFT, m, n,
	             d_VH,m,
	             d_S,1,
	             d_svh,m
	             );

	cudaDeviceSynchronize();
	//cublasDestroy(handle);	 	
}
void matmul(dt *d_r,dt *d_A,int m,int d,int n,cublasHandle_t handle)
{
	//d_r m*m   d_A m*d*nz

	dt alpha = 1.0;
	dt beta = 0.0;

	dim3 threads(1024,1,1);
	dim3 block0((m*d*n+1024-1)/1024,1,1);
	dt *d_A_mode3,*d_A_mode3T;
	cudaMalloc((void**)&d_A_mode3,sizeof(dt)*m*d*n);
	cudaMalloc((void**)&d_A_mode3T,sizeof(dt)*m*d*n);
	tensorToMode3<<<block0,threads>>>(d_A,d_A_mode3,m,d,n);
	cudaDeviceSynchronize();
	transpose<<<block0,threads>>>(d_A_mode3,d_A_mode3T,n,d*m);
	cudaDeviceSynchronize();

	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
	            m,d*n,m,
	            &alpha,d_r,m,d_A,m,
	            &beta,d_A,m
	            );
	cudaDeviceSynchronize();

	cudaFree(d_A_mode3);
	cudaFree(d_A_mode3T);

}
void division(dt *d_A,dt norm_s,dt *d_r,int l,int d, int r,cublasHandle_t handle)
{
	dt alpha = norm_s;
	dt beta = 0.0;
	cublasSgeam(handle,
	            CUBLAS_OP_N,CUBLAS_OP_N,l,r*d,
	            &alpha,d_A,l,
	            &beta,d_A,l,d_r,l
	            );

	cudaDeviceSynchronize();

}

__global__ void together(float *d_A,float *d_des,int a,int b,int c)
{
  int i = blockDim.x*blockIdx.x+threadIdx.x;  // number of threads
  const long long temp = blockDim.x*gridDim.x;
  while(i<a*b*c)
  {
  	d_A[i] = d_des[i];
  	i+=temp;
  }
__syncthreads();
}

__global__ void mode2(dt *A,dt *B,long m,long n,long r)
{
	long long i = blockIdx.x*blockDim.x+threadIdx.x;
	const long long temp = blockDim.x*gridDim.x;
	while(i<m*r*n){
		long long row=i/n;
		long long col = i%n;
		long long ge = i/(m*n);
		B[i]=A[(row-ge*m)+(col*m+ge*m*n)];		
		i+=temp;
	}
	__syncthreads();	
}
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

void ncon(dt *d_L,dt *d_M,dt *d_A,int *sizeA,int *sizeL,dt *d_out,cublasHandle_t handle,cutensorHandle_t tensor_handle)
{
	//cout<<"okay!"<<endl;
	/***************************************
	* L 的size：sizeL[0] * sizeL[1] *sizeL[2]
	* M 的size：2*2*4*4
	* A 的size ：sizeA[0] * sizeA[1] *sizeA[2]
	*收缩的顺序（调整之后）:
	* A (L M) A
	**************************************/
	dt *d_LM;
	dim3 threads(1024,1,1);
	dim3 blockA((sizeA[0]*sizeA[1]*sizeA[2]+1024-1)/1024,1,1);
	dim3 blockL((sizeL[0]*sizeL[1]*sizeL[2]+1024-1)/1024,1,1);
	dim3 blockM((2*2*4*4+1024-1)/1024,1,1);

	dt *d_L_mode3,*d_M_mode3,*d_A_mode3;
	cudaMalloc((void**)&d_L_mode3,sizeof(dt)*sizeL[0]*sizeL[1]*sizeL[2]);
	cudaMalloc((void**)&d_M_mode3,sizeof(dt)*2*2*4*4);
	cudaMalloc((void**)&d_A_mode3,sizeof(dt)*sizeA[0]*sizeA[1]*sizeA[2]);

	dt *d_L_mode3T,*d_M_mode3T,*d_A_mode3T;
	cudaMalloc((void**)&d_L_mode3T,sizeof(dt)*sizeL[0]*sizeL[1]*sizeL[2]);
	cudaMalloc((void**)&d_M_mode3T,sizeof(dt)*2*2*4*4);
	cudaMalloc((void**)&d_A_mode3T,sizeof(dt)*sizeA[0]*sizeA[1]*sizeA[2]);


	tensorToMode3<<<blockL,threads>>>(d_L,d_L_mode3,sizeL[0],sizeL[1],sizeL[2]);
	cudaDeviceSynchronize();
	transpose<<<blockL,threads>>>(d_L_mode3,d_L_mode3T,sizeL[2],sizeL[0]*sizeL[1]);
	cudaDeviceSynchronize();

	tensorToMode3<<<blockM,threads>>>(d_M,d_M_mode3,2*2,4,4);
	cudaDeviceSynchronize();
	transpose<<<blockM,threads>>>(d_M_mode3,d_M_mode3T,4,2*2*4);
	cudaDeviceSynchronize();
	tensorToMode3<<<blockA,threads>>>(d_A,d_A_mode3,sizeA[0],sizeA[1],sizeA[2]);
	cudaDeviceSynchronize();
	transpose<<<blockA,threads>>>(d_A_mode3,d_A_mode3T,sizeA[2],sizeA[0]*sizeA[1]);
	cudaDeviceSynchronize();

	
	cudaMalloc((void**)&d_LM,sizeof(dt)*sizeL[0] *sizeL[1]*2*2*4);
	dt alpha = 1.0;
	dt beta = 0.0;
	
	cublasSgemm(handle,
	            CUBLAS_OP_N,CUBLAS_OP_T,
	            sizeL[0] * sizeL[1],2*2*4,4,
	            &alpha,d_L,sizeL[0] * sizeL[1],d_M,2*2*4,
	            &beta,d_LM,sizeL[0] * sizeL[1]
	            );
	            
	
	cudaDeviceSynchronize();

	//[2,4,-2][1,2,4,5,-3]  
	// d_LM permute ->[-3,1,5,2,4]
	// (A+)(LM)
	permute(d_LM,sizeL[0],sizeL[1],2,2,4,tensor_handle,0);
	cudaDeviceSynchronize();

	dt *d_ALM;	
	cudaMalloc((void**)&d_ALM,sizeof(dt)*sizeA[2]*4*sizeA[0]*2);	
	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,
	            sizeA[2],4*sizeA[0]*2,sizeA[0]*sizeA[1],
	            &alpha,d_A,sizeA[0]*sizeA[1],d_LM,sizeA[0]*sizeA[1]*4,
	            &beta,d_ALM,sizeA[2]
	            );	            
	// A * ALM 
	//[1,5,-1][-2,-3,1,5]
	 cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,
  	            sizeA[2],sizeA[2]*4,sizeA[0]*sizeA[1],
  	            &alpha,d_A,sizeA[0]*sizeA[1],d_ALM,sizeA[2]*4,
  	            &beta,d_out,sizeA[2]
  	            ); 	
  	cudaDeviceSynchronize();

  	cudaFree(d_LM);
  	cudaFree(d_ALM);
  	cudaFree(d_L_mode3);
  	cudaFree(d_M_mode3);
  	cudaFree(d_A_mode3);
  	cudaFree(d_L_mode3T);
  	cudaFree(d_M_mode3T);
  	cudaFree(d_A_mode3T);
}
float cuda_dot(dt *A,dt *B,int n,dt result,cublasHandle_t handle)
{
  //cublasHandle_t handle;
  //cublasCreate(&handle);
  cublasSdot(handle,n,A,1,B,1,&result);
  cudaDeviceSynchronize();
  return result;
}
void eig(dt *d_A,dt *d_W,int n)
{

 	cusolverDnHandle_t cusolverH;
 	cusolverDnCreate(&cusolverH);
	 cudaStream_t stream = NULL;
	 cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	 cusolverDnSetStream(cusolverH, stream);

    syevjInfo_t syevj_params = NULL;
    int *d_info = NULL; /* error info */
    int  lwork = 0;
    dt *d_work = NULL;
    dt tol = 1.e-7;
    int max_sweeps = 15;
 	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
 	cublasFillMode_t  uplo = CUBLAS_FILL_MODE_LOWER;

 	cusolverDnCreateSyevjInfo(&syevj_params);
 	cusolverDnXsyevjSetTolerance(syevj_params,tol);
 	cusolverDnXsyevjSetMaxSweeps(syevj_params,max_sweeps);
 	 	
 	cudaMalloc ((void**)&d_info, sizeof(int));
 	
 	cusolverDnSsyevj_bufferSize(cusolverH,jobz,uplo, 
        n,d_A,n,d_W, &lwork,syevj_params);

 	cudaMalloc((void**)&d_work, sizeof(dt)*lwork);
 	cusolverDnSsyevj(cusolverH,jobz,uplo, 
        n,d_A,n,d_W, d_work,lwork,d_info,syevj_params);

 	cudaDeviceSynchronize();
 	cudaFree(d_info);
 	cudaFree(d_work);
 	cudaStreamDestroy(stream);
 	cusolverDnDestroySyevjInfo(syevj_params);
	
}
void ncon2(dt *d_R,dt *d_M,dt *d_A,int *sizeA,int *sizeR,dt *d_out,cublasHandle_t handle,cutensorHandle_t tensor_handle)
{
	
	//cout<<"-------okay2---------"<<endl;
	dt alpha = 1.0;
	dt beta = 0.0;

	dim3 threads(1024,1,1);
	dim3 blockA((sizeA[0]*sizeA[1]*sizeA[2]+1024-1)/1024,1,1);
	dim3 blockR((sizeR[0]*sizeR[1]*sizeR[2]+1024-1)/1024,1,1);
	dim3 blockM((2*2*4*4+1024-1)/1024,1,1);
	dim3 block0((sizeA[0]*sizeA[1]*sizeR[1]*sizeR[2]+1024-1)/1024,1,1);

	dt *d_R_mode3,*d_M_mode3,*d_A_mode3;
	cudaMalloc((void**)&d_R_mode3,sizeof(dt)*sizeR[0]*sizeR[1]*sizeR[2]);
	cudaMalloc((void**)&d_M_mode3,sizeof(dt)*2*2*4*4);
	cudaMalloc((void**)&d_A_mode3,sizeof(dt)*sizeA[0]*sizeA[1]*sizeA[2]);

	dt *d_R_mode3T,*d_M_mode3T,*d_A_mode3T;
	cudaMalloc((void**)&d_R_mode3T,sizeof(dt)*sizeR[0]*sizeR[1]*sizeR[2]);
	cudaMalloc((void**)&d_M_mode3T,sizeof(dt)*2*2*4*4);
	cudaMalloc((void**)&d_A_mode3T,sizeof(dt)*sizeA[0]*sizeA[1]*sizeA[2]);


	tensorToMode3<<<blockR,threads>>>(d_R,d_R_mode3,sizeR[0],sizeR[1],sizeR[2]);
	cudaDeviceSynchronize();
	transpose<<<blockR,threads>>>(d_R_mode3,d_R_mode3T,sizeR[2],sizeR[0]*sizeR[1]);
	cudaDeviceSynchronize();

	tensorToMode3<<<blockM,threads>>>(d_M,d_M_mode3,2*2,4,4);
	cudaDeviceSynchronize();
	transpose<<<blockM,threads>>>(d_M_mode3,d_M_mode3T,4,2*2*4);
	cudaDeviceSynchronize();
	tensorToMode3<<<blockA,threads>>>(d_A,d_A_mode3,sizeA[0],sizeA[1],sizeA[2]);
	cudaDeviceSynchronize();
	transpose<<<blockA,threads>>>(d_A_mode3,d_A_mode3T,sizeA[2],sizeA[0]*sizeA[1]);
	cudaDeviceSynchronize();


	dt *d_AR;
	cudaMalloc((void**)&d_AR,sizeof(dt)*sizeA[0]*sizeA[1]*sizeR[1]*sizeR[2]);

	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
	            sizeA[0]*sizeA[1],sizeR[1]*sizeR[2],sizeA[2],
	            &alpha,d_A,sizeA[0]*sizeA[1],d_R,sizeA[2],
	            &beta,d_AR,sizeA[0]*sizeA[1]
	            );

	//[-1,1,3,4] -> [-1,1,4,3]
	//(-1,1),3,4 ___mode2___3,[(-1,1),4]___转置___[(-1,1),4],3 
	dt *d_AR_mode2,*d_ARA;
	cudaMalloc((void**)&d_AR_mode2,sizeof(dt)*sizeA[0]*sizeA[1]*sizeR[1]*sizeR[2]);
	cudaMalloc((void**)&d_ARA,sizeof(dt)*sizeA[0]*sizeA[1]*sizeR[2]*sizeA[0]*sizeA[1]);
	mode2<<<block0,threads>>>(d_AR,d_AR_mode2,sizeA[0]*sizeA[1],sizeR[1],sizeR[2]);
	cudaDeviceSynchronize();

	cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_T,
	            sizeA[0]*sizeA[1]*sizeR[2],sizeA[0]*sizeA[1],sizeR[1],
	            &alpha,d_AR_mode2,sizeR[1],d_A,sizeA[0]*sizeA[1],
	            &beta,d_ARA,sizeA[0]*sizeA[1]*sizeR[2]
	            );
	

	//(-1,1,4,-2,5) -> (-1,-2,5,1,4)
	permute(d_ARA,sizeA[0],sizeA[1],sizeR[2],sizeA[0],sizeA[1],tensor_handle,1);
	

  		
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
	            sizeA[0]*sizeA[0],4,2*2*4,
	            &alpha,d_ARA,sizeA[0]*sizeA[0],d_M,2*2*4,
	            &beta,d_out,sizeA[0]*sizeA[0]
	            );
		
	cudaFree(d_AR);
	cudaFree(d_ARA);
	cudaFree(d_AR_mode2);
	cudaFree(d_R_mode3);
  	cudaFree(d_M_mode3);
  	cudaFree(d_A_mode3);
  	cudaFree(d_R_mode3T);
  	cudaFree(d_M_mode3T);
  	cudaFree(d_A_mode3T);
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
