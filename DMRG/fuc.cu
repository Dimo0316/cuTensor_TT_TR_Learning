#include "head.h"


void QR(dt *d_A,dt *d_R,int m,int n,cusolverDnHandle_t cusolverH)
{
	//dim3 threads(1024,1,1);
	//dim3 block0((m+1024-1)/1024,1,1);

	dt *d_tau = NULL;
	int *devInfo = NULL;
	dt *d_work = NULL;
	int lwork_geqrf = 0;
	int lwork_orgqr = 0;
	int lwork = 0;

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
	upper<<<100,1024>>>(d_A,d_R,m,n);
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
	Init_P<<<100,100>>>(d_P,a);
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
/*
	cudaMemcpy(p_Q,d_Q,sizeof(dt)*a*b*c,cudaMemcpyDeviceToHost);
	cout<<"RQ decomposition Q:"<<endl;
	for(int i=0;i<a*b*c;i++)
	{
		cout<<p_Q[i]<<" _ ";
	}
	cout<<endl;
	cout<<"*********************************"<<endl;
*/
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
/*
	cudaMemcpy(p_R,d_QRweight,sizeof(dt)*a*a,cudaMemcpyDeviceToHost);
	cout<<"RQ decomposition R:"<<endl;
	for(int i=0;i<a*a;i++)
	{
		cout<<p_R[i]<<" _ ";
	}
	cout<<endl;
*/
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
	//d_r m*m   d_A m*d*n

	dt alpha = 1.0;
	dt beta = 0.0;
	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
	            m,d*n,m,
	            &alpha,d_r,m,d_A,m,
	            &beta,d_A,m
	            );
	cudaDeviceSynchronize();

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
void ncon_1(float *A_d,float *B_d,float *C_d,vector<int> modeA,vector<int> modeB,vector<int> modeC,unordered_map<int, int64_t> extent,cutensorHandle_t handle)
{
    typedef float floatTypeA;
    typedef float floatTypeB;
    typedef float floatTypeC;
    typedef float floatTypeCompute;

    cudaDataType_t typeA = CUDA_R_32F;
    cudaDataType_t typeB = CUDA_R_32F;
    cudaDataType_t typeC = CUDA_R_32F;
    cutensorComputeType_t typeCompute = CUTENSOR_R_MIN_32F;

    floatTypeCompute alpha = (floatTypeCompute)1.1f;
    floatTypeCompute beta  = (floatTypeCompute)0.0f;

    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();

   vector<int64_t> extentC;
    for (auto mode : modeC)
        extentC.push_back(extent[mode]);
    vector<int64_t> extentA;
    for (auto mode : modeA)
        extentA.push_back(extent[mode]);
    vector<int64_t> extentB;
    for (auto mode : modeB)
        extentB.push_back(extent[mode]);

    size_t elementsA = 1;
    for (auto mode : modeA)
        elementsA *= extent[mode];
    size_t elementsB = 1;
    for (auto mode : modeB)
        elementsB *= extent[mode];
    size_t elementsC = 1;
    for (auto mode : modeC)
        elementsC *= extent[mode];

    cutensorTensorDescriptor_t descA;
    cutensorInitTensorDescriptor(&handle,
                 &descA,
                 nmodeA,
                 extentA.data(),
                 NULL,/*stride*/
                 typeA, CUTENSOR_OP_IDENTITY);

    cutensorTensorDescriptor_t descB;
   cutensorInitTensorDescriptor(&handle,
                 &descB,
                 nmodeB,
                 extentB.data(),
                 NULL,/*stride*/
                 typeB, CUTENSOR_OP_IDENTITY);

    cutensorTensorDescriptor_t descC;
   cutensorInitTensorDescriptor( &handle,
                 &descC,
                 nmodeC,
                 extentC.data(),
                 NULL,/*stride*/
                 typeC, CUTENSOR_OP_IDENTITY);

     uint32_t alignmentRequirementA;
     cutensorGetAlignmentRequirement(&handle,
                  A_d,
                  &descA,
                  &alignmentRequirementA);

     uint32_t alignmentRequirementB;
     cutensorGetAlignmentRequirement(&handle,
                  B_d,
                  &descB,
                  &alignmentRequirementB);

     uint32_t alignmentRequirementC;
     cutensorGetAlignmentRequirement(&handle,
                  C_d,
                  &descC, 
                  &alignmentRequirementC);

    cutensorContractionDescriptor_t desc;
   cutensorInitContractionDescriptor(&handle, 
                 &desc,
                 &descA, modeA.data(), alignmentRequirementA,
                 &descB, modeB.data(), alignmentRequirementB,
                 &descC, modeC.data(), alignmentRequirementC,
                 &descC, modeC.data(), alignmentRequirementC,
                 typeCompute);

    cutensorContractionFind_t find;
    cutensorInitContractionFind( 
                 &handle, &find, 
                 CUTENSOR_ALGO_DEFAULT);

    uint64_t worksize = 0;
    cutensorContractionGetWorkspace(&handle,
                 &desc,
                 &find,
                 CUTENSOR_WORKSPACE_RECOMMENDED, &worksize);

    void *work = nullptr;
    if (worksize > 0)
    {
        if (cudaSuccess != cudaMalloc(&work, worksize))
        {
            work = nullptr;
            worksize = 0;
        }
    } 

    cutensorContractionPlan_t plan;
    cutensorInitContractionPlan(&handle,
                 &plan,
                 &desc,
                 &find,
                 worksize);

    cutensorStatus_t err;
    err = cutensorContraction(&handle,
                                  &plan,
                                  (void*) &alpha, A_d, B_d,
                                  (void*) &beta,  C_d, C_d, 
                                  work, worksize, 0 /* stream */);


        if (err != CUTENSOR_STATUS_SUCCESS)
        {
            cout<<"over"<<endl; 
            printf("ERROR: %s in line %d\n", cutensorGetErrorString(err), __LINE__);
        }

    //if (work) cudaFree(work);
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
	cudaMalloc((void**)&d_LM,sizeof(dt)*sizeL[0] *sizeL[1]*2*2*4);
	vector<int> modeLM{'a','b','d','e','f'};
    vector<int> modeL{'a','b','c'};
    vector<int> modeM{'d','e','f','c'};
    unordered_map<int, int64_t> extent;
    extent['a'] = sizeL[0];
    extent['b'] = sizeL[1];
    extent['c'] = 4;
    extent['d'] = 2;
    extent['e'] = 2;
    extent['f'] = 4;
    ncon_1(d_L,d_M,d_LM,modeL,modeM,modeLM,extent,tensor_handle);
    
    dt *d_ALM;
	cudaMalloc((void**)&d_ALM,sizeof(dt)*sizeA[2]*4*sizeA[0]*2);
	vector<int> modeA{'b','d','h'};	
	vector<int> modeALM{'h','a','e','f'};
	unordered_map<int, int64_t> extent2;
	extent2['a'] = sizeL[0];
    extent2['b'] = sizeL[1];
    extent2['d'] = 2;
    extent2['e'] = 2;
    extent2['f'] = 4;
    extent2['h'] = sizeA[2];
    ncon_1(d_A,d_LM,d_ALM,modeA,modeLM,modeALM,extent2,tensor_handle);

    vector<int> modeA2{'a','e','g'};
    vector<int> modeOUT{'g','h','f'};
    unordered_map<int, int64_t> extent3;
    extent3['a'] = sizeL[0];
	extent3['e'] = 2;
	extent3['h'] = sizeA[2];
	extent3['f'] = 4;
	extent3['g'] = sizeA[2];
	ncon_1(d_A,d_ALM,d_out,modeA2,modeALM,modeOUT,extent3,tensor_handle);

	cudaFree(d_LM);
	cudaFree(d_ALM);

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


	dt *d_AR;
	cudaMalloc((void**)&d_AR,sizeof(dt)*sizeA[0]*sizeA[1]*sizeR[1]*sizeR[2]);
	vector<int> modeAR{'a','b','d','e'};
    vector<int> modeA{'a','b','c'};
    vector<int> modeR{'c','d','e'};
    unordered_map<int, int64_t> extent;
    extent['a'] = sizeA[0];
    extent['b'] = sizeA[1];
    extent['c'] = sizeA[2];
    extent['d'] = sizeR[1];
    extent['e'] = sizeR[2];    
    ncon_1(d_A,d_R,d_AR,modeA,modeR,modeAR,extent,tensor_handle);
    
    dt *d_ARA;
	cudaMalloc((void**)&d_ARA,sizeof(dt)*sizeA[0]*sizeA[1]*sizeR[2]*sizeA[0]*sizeA[1]);	
	vector<int> modeA2{'f','g','d'};
	vector<int> modeARA{'a','b','e','f','g'};
	unordered_map<int, int64_t> extent2;
	extent2['a'] = sizeA[0];
    extent2['b'] = sizeA[1];
    extent2['d'] = sizeR[1];
    extent2['e'] = sizeR[2];
    extent2['f'] = sizeA[0];
    extent2['g'] = sizeA[1];
    ncon_1(d_AR,d_A,d_ARA,modeAR,modeA2,modeARA,extent2,tensor_handle);

    vector<int> modeM{'g','b','e','h'};
    vector<int> modeOUT{'a','f','h'};
    unordered_map<int, int64_t> extent3;
    extent3['a'] = sizeA[0];
	extent3['b'] = sizeA[1];
	extent3['e'] = sizeR[2];
	extent3['f'] = sizeA[0];
	extent3['g'] = sizeA[1];
	extent3['h'] = 4;
	ncon_1(d_ARA,d_M,d_out,modeARA,modeM,modeOUT,extent3,tensor_handle);


	cudaFree(d_AR);
	cudaFree(d_ARA);
	
}