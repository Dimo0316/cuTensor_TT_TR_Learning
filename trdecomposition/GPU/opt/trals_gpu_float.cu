#include "head.h"


void trals_gpu_float(float* h_tensor, int* h_n_array, int* h_r_array, double* eps, int maxiter=10){
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    cudaError_t cudaStat5 = cudaSuccess;
    cudaError_t cudaStat6 = cudaSuccess;
    
 	  dim3 threads(512,1,1);
    dt alpha = 1.0;
    dt beta = 0.0;
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;

    cublasHandle_t cublasH;
    cublas_status = cublasCreate(&cublasH);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    cusolverDnHandle_t cusolverH = NULL;
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
    if(algo == CUBLAS_GEMM_DEFAULT)
        cublasSetMathMode(cublasH, CUBLAS_DEFAULT_MATH);
    else if( algo == CUBLAS_GEMM_DEFAULT_TENSOR_OP)
        cublasSetMathMode(cublasH, CUBLAS_TENSOR_OP_MATH);

    long h_num_tensor = h_n_array[0]*h_n_array[1]*h_n_array[2];
    long h_num_tensor01 = h_n_array[0]*h_n_array[1];
    long h_num_tensor12 = h_n_array[1]*h_n_array[2];
    long h_num_tensor02 = h_n_array[2]*h_n_array[0];
    long h_num_core1 = h_r_array[0]*h_n_array[0]*h_r_array[1];
    long h_num_core2 = h_r_array[1]*h_n_array[1]*h_r_array[2];
    long h_num_core3 = h_r_array[2]*h_n_array[2]*h_r_array[0];
    long h_num_core01 = h_r_array[0]*h_r_array[1];
    long h_num_core12 = h_r_array[1]*h_r_array[2];
    long h_num_core02 = h_r_array[2]*h_r_array[0];

    //创建空间 d_tensor 和三个 TR核
    dt *d_tensor, *d_core1,*d_core2,*d_core3;
    cudaStat1 = cudaMalloc((void**)&d_tensor,sizeof(dt)*h_num_tensor);
    cudaStat3 = cudaMalloc((void**)&d_core1,sizeof(dt)*h_num_core1);
    cudaStat4 = cudaMalloc((void**)&d_core2,sizeof(dt)*h_num_core2);
    cudaStat5 = cudaMalloc((void**)&d_core3,sizeof(dt)*h_num_core3);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);
    assert(cudaStat4 == cudaSuccess);

    int lwork=0;
	  dt *d_work = NULL;
    int *d_Ipiv = NULL, *d_info = NULL; // PA=LU, P is control weather permute
	  cudaMalloc((void**)&d_Ipiv,sizeof(int));
	  cudaMalloc((void**)&d_info,sizeof(int));

    //初始化 d_tensor 和三个 TR核, 根据计算的特性，TR张量的第一个核不需要初始化
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen,1234ULL);
    curandGenerateUniform(gen,d_core2,h_num_core2);
    curandGenerateUniform(gen,d_core3,h_num_core3);
    cudaStat1 = cudaMemcpyAsync(d_tensor, h_tensor,sizeof(dt)*h_num_tensor,cudaMemcpyHostToDevice,0);
    assert(cudaStat1 == cudaSuccess);
    // printTensor(d_tensor,h_n_array[0],h_n_array[1],h_n_array[2],"tensor");
    //cudaMemcpyAsync(d_B,B,sizeof(dt)*b*r,cudaMemcpyHostToDevice,0);
    //cudaMemcpyAsync(d_C,C,sizeof(dt)*c*r,cudaMemcpyHostToDevice,0);
    // printTensor(d_core2, h_r_array[1], h_n_array[1], h_r_array[2], "core2");
    // printTensor(d_core3, h_r_array[2], h_n_array[2], h_r_array[0], "core3");

/// coveroing tr-als 开始迭代最小二乘   A x' = B'
for(int i = 0;i<maxiter;i++){

/// 1) 获得core1
    float *d_ATA = NULL, *d_core23T231 = NULL,*d_ATBT = NULL, *d_core23 = NULL, *d_core1T213=NULL;
    cudaStat1 = cudaMalloc((void**)&d_ATA, sizeof(float)*h_num_core01*h_num_core01);
    cudaStat2 = cudaMalloc((void**)&d_ATBT, sizeof(float)*h_num_core01*h_n_array[0]);
    cudaStat3 = cudaMalloc((void**)&d_core23T231,sizeof(dt)*h_r_array[1]*h_num_tensor12*h_r_array[0]);
    cudaStat4 = cudaMalloc((void**)&d_core23,sizeof(dt)*h_r_array[1]*h_num_tensor12*h_r_array[0]);
    cudaStat5 = cudaMalloc((void**)&d_core1T213,sizeof(dt)*h_num_core1);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);
    assert(cudaStat4 == cudaSuccess);
    assert(cudaStat5 == cudaSuccess);
    // printMatrix_Device( h_n_array[0], h_num_tensor12, d_tensor, h_n_array[0], "T1");
    // cublas_status = cublasSgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
    //                       h_num_tensor12, h_n_array[0],
    //                       &alpha,
    //                       d_tensor, h_n_array[0],
    //                       &beta,
    //                       d_tensorT, h_num_tensor12,
    //                       d_tensorT, h_num_tensor12);
    // assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    // printMatrix_Device(h_num_tensor12, h_n_array[0], d_tensorT, h_num_tensor12, "T1T");
    // 1. 计算左矩阵A  d_core23T231
    cublas_status = cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                           h_r_array[1]*h_n_array[1], //r2*n2
                           h_n_array[2]*h_r_array[0], //n2*r3
                           h_r_array[2],  //r3
                           &alpha, d_core2, CUDA_R_32F, h_r_array[1]*h_n_array[1], 
                           d_core3, CUDA_R_32F, h_r_array[2],
                           &beta, d_core23, CUDA_R_32F, h_r_array[1]*h_n_array[1],
                           CUDA_R_32F, algo);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
	  dim3 block_core23T231((h_r_array[1]*h_num_tensor12*h_r_array[0]+512-1)/512,1,1); //for 231
	// printf("%d  %d %d\n",  h_num_tensor12, h_r_array[1], h_r_array[0]);
    // printTensor(d_core23,  h_r_array[1], h_num_tensor12, h_r_array[0], "core23");
    tensorToMode231<<<block_core23T231,threads>>>(d_core23, d_core23T231, h_r_array[1],h_num_tensor12,h_r_array[0]);
    cudaFree(d_core23);
    // printTensor(d_core23T231, h_num_tensor12, h_r_array[0], h_r_array[1], "core23T231");
    // 计算ATA
    // printMatrix_Device(h_num_tensor12,h_num_core01,d_core23T231,h_num_tensor12,"A1");
    // printMatrix_Device(h_n_array[0],h_num_tensor12,d_tensor,h_n_array[0],"tensor");
    cublas_status = cublasGemmEx(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                           h_num_core01, //r2*n2
                           h_num_core01, //n2*r3
                           h_num_tensor12,  //r3
                           &alpha, d_core23T231, CUDA_R_32F, h_num_tensor12, 
                           d_core23T231, CUDA_R_32F, h_num_tensor12,
                           &beta, d_ATA, CUDA_R_32F, h_num_core01,
                           CUDA_R_32F, algo);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    // printMatrix_Device(h_num_core01,h_num_core01,d_ATA,h_num_core01,"ATA");
    // 2. 计算右矩阵ATBT = ATBT  
    cublas_status = cublasGemmEx(cublasH, CUBLAS_OP_T, CUBLAS_OP_T,
                           h_num_core01, //r2*n2
                           h_n_array[0], //n2*r3
                           h_num_tensor12,  //r3
                           &alpha, d_core23T231, CUDA_R_32F, h_num_tensor12, 
                           d_tensor, CUDA_R_32F, h_n_array[0],
                           &beta, d_ATBT, CUDA_R_32F, h_num_core01,
                           CUDA_R_32F, algo);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    // printMatrix_Device(h_num_core01,h_n_array[0],d_ATBT,h_num_core01,"ATBT");
    cudaFree(d_core23T231);
    
    // 3. 利用线性求解获得 d_core' 并获得d_core
    cusolver_status = cusolverDnSgetrf_bufferSize(cusolverH,h_num_core01,h_num_core01,d_ATA,h_num_core01,&lwork);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
	cudaStat1 = cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
	assert(cudaStat1 == cudaSuccess);
	// printf("lwork1: %d  h_num_core01: %d  \n", lwork, h_num_core12);
	cusolver_status = cusolverDnSgetrf(cusolverH,h_num_core01,h_num_core01,d_ATA,h_num_core01,d_work,d_Ipiv,d_info);
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
	cusolver_status =cusolverDnSgetrs(cusolverH,CUBLAS_OP_N,h_num_core01,h_n_array[0],d_ATA,h_num_core01,d_Ipiv,d_ATBT,h_num_core01,d_info);
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
	cudaStat1 = cudaDeviceSynchronize();
	assert(cudaStat1 == cudaSuccess);
	// printMatrix_Device(h_num_core01, h_n_array[0], d_ATBT, h_num_core01, "ATBTT");

	cublas_status = cublasSgeam(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,h_n_array[0],h_num_core01,&alpha,d_ATBT,h_num_core01,&beta,d_core1,h_n_array[0],d_core1T213,h_n_array[0]);
	assert(cublas_status == CUBLAS_STATUS_SUCCESS);
	// printMatrix_Device(h_n_array[0], h_num_core01, d_core1T213, h_n_array[0], "core1T213Matrix");
	// printTensor(d_core1T213, h_n_array[0], h_r_array[0], h_r_array[1], "core1T213");
	dim3 block_core1T213((h_num_core1+512-1)/512,1,1); //for core1T213
	tensorToMode2<<<block_core1T213,threads>>>(d_core1T213,d_core1,h_n_array[0],h_r_array[0],h_r_array[1]);
	// printTensor(d_core1, h_r_array[0], h_n_array[0] ,h_r_array[1], "core1");
	cudaFree(d_core1T213);
    cudaFree(d_ATBT);
    cudaFree(d_ATA);


    printf("%ld\n", sizeof(dt)*h_num_tensor02*h_num_core12);
/// 2) 获得core2   A'*A*X = A'*B 
	//获得A
	float *d_core31 = NULL, *d_core31T231 = NULL, *d_ATA2 = NULL, *d_tensorT231=NULL, *d_ATBT2=NULL, *d_core2T213 = NULL;
	cudaStat1 = cudaMalloc((void**)&d_core31,sizeof(dt)*h_num_tensor02*h_num_core12);
	cudaStat2 = cudaMalloc((void**)&d_core31T231,sizeof(dt)*h_num_tensor02*h_num_core12);
	cudaStat3 = cudaMalloc((void**)&d_ATA2, sizeof(float)*h_num_tensor12*h_num_core12);
	cudaStat4 = cudaMalloc((void**)&d_tensorT231, sizeof(float)*h_num_tensor);
	cudaStat5 = cudaMalloc((void**)&d_ATBT2, sizeof(float)*h_n_array[1]*h_num_core12);
	cudaStat6 = cudaMalloc((void**)&d_core2T213, sizeof(float)*h_num_core2);
	assert(cudaStat1 == cudaSuccess);
	assert(cudaStat2 == cudaSuccess);
	assert(cudaStat3 == cudaSuccess);
	assert(cudaStat4 == cudaSuccess);
	assert(cudaStat5 == cudaSuccess);
	assert(cudaStat6 == cudaSuccess);

	// printTensor(d_core1, h_r_array[0], h_n_array[0] ,h_r_array[1], "core1");
	// printTensor(d_core3, h_r_array[2], h_n_array[2] ,h_r_array[0], "core3");
	int h_num_n3r3 = h_r_array[2]*h_n_array[2];
	cublas_status = cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                           h_num_n3r3,//h_r_array[2]*h_n_array[2], //r3*n3
                           h_n_array[0]*h_r_array[1], //n1*r2
                           h_r_array[0],  //r1
                           &alpha, d_core3, CUDA_R_32F, h_num_n3r3, 
                           d_core1, CUDA_R_32F, h_r_array[0],
                           &beta, d_core31, CUDA_R_32F, h_num_n3r3,
                           CUDA_R_32F, algo);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
	dim3 block_core31T231((h_num_tensor02*h_num_core12+512-1)/512,1,1); //for 231
    // printTensor(d_core31,  h_r_array[2], h_num_tensor02, h_r_array[1], "core31");
    tensorToMode231<<<block_core31T231,threads>>>(d_core31, d_core31T231, h_r_array[2], h_num_tensor02, h_r_array[1]);
    cudaFree(d_core31);
	// printTensor(d_core31T231, h_num_tensor02, h_r_array[1], h_r_array[2], "core31T231");
	// printMatrix_Device(h_num_tensor02, h_num_core12,d_core31T231, h_num_tensor02,"A2");

	//计算ATA2
	cublas_status = cublasGemmEx(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                           h_num_core12, //r2*r3
                           h_num_core12, //r2*r3
                           h_num_tensor02,  //n1*n3
                           &alpha, d_core31T231, CUDA_R_32F, h_num_tensor02, 
                           d_core31T231, CUDA_R_32F, h_num_tensor02,
                           &beta, d_ATA2, CUDA_R_32F, h_num_core12,
                           CUDA_R_32F, algo);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    // printMatrix_Device(h_num_core12,h_num_core12,d_ATA2,h_num_core12,"ATA2");
    dim3 block_tensorT231((h_num_tensor+512-1)/512,1,1); //for core1T213
	tensorToMode231<<<block_tensorT231,threads>>>(d_tensor, d_tensorT231, h_n_array[0],h_n_array[1],h_n_array[2]);
	// printMatrix_Device(h_n_array[1],h_num_tensor02,d_tensorT231,h_n_array[1],"tensorT231");
    //计算ATBT2
    cublas_status = cublasGemmEx(cublasH, CUBLAS_OP_T, CUBLAS_OP_T,
                           h_num_core12, //r2*n2
                           h_n_array[1], //n2*r3
                           h_num_tensor02,  //r3
                           &alpha, d_core31T231, CUDA_R_32F, h_num_tensor02, 
                           d_tensorT231, CUDA_R_32F, h_n_array[1],
                           &beta, d_ATBT2, CUDA_R_32F, h_num_core12,
                           CUDA_R_32F, algo);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    // printMatrix_Device(h_num_core12, h_n_array[1], d_ATBT2, h_num_core12, "ATBT2");
	cudaFree(d_core31T231);
	cudaFree(d_tensorT231);
	//求解线性方程
	cusolver_status = cusolverDnSgetrf_bufferSize(cusolverH,h_num_core12,h_num_core12,d_ATA2,h_num_core12,&lwork);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
	cudaStat1 = cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
	assert(cudaStat1 == cudaSuccess);
	// printf("lwork: %d  h_num_core12: %d  \n", lwork, h_num_core12);
	cusolver_status = cusolverDnSgetrf(cusolverH,h_num_core12,h_num_core12,d_ATA2,h_num_core12,d_work,d_Ipiv,d_info);
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
	cudaStat1 = cudaDeviceSynchronize();
	assert(cudaStat1 == cudaSuccess);
	cusolver_status =cusolverDnSgetrs(cusolverH,CUBLAS_OP_N,h_num_core12,h_n_array[1],d_ATA2,h_num_core12,d_Ipiv,d_ATBT2,h_num_core12,d_info);
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
	cudaStat1 = cudaDeviceSynchronize();
	assert(cudaStat1 == cudaSuccess);
	cudaFree(d_ATA2);
	// printMatrix_Device(h_num_core12, h_n_array[1], d_ATBT2, h_num_core12, "ATBT2T");

	cublas_status = cublasSgeam(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,h_n_array[1],h_num_core12,&alpha,d_ATBT2,h_num_core12,&beta,d_core2T213,h_n_array[1],d_core2T213,h_n_array[1]);
	assert(cublas_status == CUBLAS_STATUS_SUCCESS);
	cudaFree(d_ATBT2);
	// printMatrix_Device(h_n_array[1], h_num_core12, d_core2T213, h_n_array[1], "core2T213Matrix");
	// printTensor(d_core2T213, h_n_array[1], h_r_array[1], h_r_array[2], "core2T213");
	dim3 block_core2T213((h_num_core2+512-1)/512,1,1); //for core1T213
	tensorToMode2<<<block_core2T213,threads>>>(d_core2T213, d_core2, h_n_array[1], h_r_array[1], h_r_array[2]);
	// printTensor(d_core2, h_r_array[1], h_n_array[1] ,h_r_array[2], "core2");
	cudaFree(d_core2T213);



/// 3) 获得core3   A'*A*X = A'*B
	float *d_core12 = NULL, *d_core12T231 = NULL, *d_ATA3 = NULL, *d_tensorT312=NULL, *d_ATBT3=NULL, *d_core3T213 = NULL;//, *d_work2 = NULL;
	cudaStat1 = cudaMalloc((void**)&d_core12,sizeof(dt)*h_num_tensor01*h_num_core02);
	cudaStat2 = cudaMalloc((void**)&d_core12T231,sizeof(dt)*h_num_tensor01*h_num_core02);
	cudaStat3 = cudaMalloc((void**)&d_ATA3,sizeof(dt)*h_num_core02*h_num_core02);
	cudaStat4 = cudaMalloc((void**)&d_tensorT312, sizeof(float)*h_num_tensor);
	cudaStat5 = cudaMalloc((void**)&d_ATBT3, sizeof(float)*h_n_array[2]*h_num_core02);
	cudaStat6 = cudaMalloc((void**)&d_core3T213, sizeof(float)*h_num_core3);
	assert(cudaStat1 == cudaSuccess);
	assert(cudaStat2 == cudaSuccess);
	assert(cudaStat3 == cudaSuccess);
	assert(cudaStat4 == cudaSuccess);
	assert(cudaStat5 == cudaSuccess);
	assert(cudaStat6 == cudaSuccess);
	//计算 A = core1*core2
	// printTensor(d_core1, h_r_array[0], h_n_array[0] ,h_r_array[1], "core1");
	// printTensor(d_core2, h_r_array[1], h_n_array[1] ,h_r_array[2], "core2");
	int h_num_n0r0 = h_r_array[0]*h_n_array[0];
	cublas_status = cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                           h_num_n0r0,//h_r_array[0]*h_n_array[0], //r3*n3
                           h_n_array[1]*h_r_array[2], //n1*r2
                           h_r_array[1],  //r1
                           &alpha, d_core1, CUDA_R_32F, h_num_n0r0, 
                           d_core2, CUDA_R_32F, h_r_array[1],
                           &beta, d_core12, CUDA_R_32F, h_num_n0r0,
                           CUDA_R_32F, algo);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    // printMatrix_Device(h_num_n0r0, h_n_array[1]*h_r_array[2], d_core12, h_num_n0r0, "core12");

    // printTensor(d_core12,  h_r_array[0], h_num_tensor01, h_r_array[2], "core12");
    dim3 block_core12T231((h_num_tensor01*h_num_core02+512-1)/512,1,1); 
    tensorToMode231<<<block_core12T231,threads>>>(d_core12, d_core12T231, h_r_array[0], h_num_tensor01, h_r_array[2]);
    if(i != maxiter-1)
    	cudaFree(d_core12);
    // printTensor(d_core12T231, h_num_tensor01, h_r_array[2], h_r_array[0], "core12T231");
	// printMatrix_Device(h_num_tensor01, h_num_core02,d_core12T231, h_num_tensor01,"A3");

	//计算ATA3
	cublas_status = cublasGemmEx(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                           h_num_core02, //r1*r3
                           h_num_core02, //r1*r3
                           h_num_tensor01,  //n1*n2
                           &alpha, d_core12T231, CUDA_R_32F, h_num_tensor01, 
                           d_core12T231, CUDA_R_32F, h_num_tensor01,
                           &beta, d_ATA3, CUDA_R_32F, h_num_core02,
                           CUDA_R_32F, algo);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    // printMatrix_Device(h_num_core02,h_num_core02,d_ATA3,h_num_core02,"ATA3");

    dim3 block_tensorT312((h_num_tensor+512-1)/512,1,1); 
	tensorToMode3<<<block_tensorT312,threads>>>(d_tensor, d_tensorT312, h_n_array[0], h_n_array[1], h_n_array[2]);
	// printMatrix_Device(h_n_array[2],h_num_tensor01,d_tensorT312,h_n_array[2],"tensorT312");

	cublas_status = cublasGemmEx(cublasH, CUBLAS_OP_T, CUBLAS_OP_T,
                           h_num_core02, //r2*n2
                           h_n_array[2], //n2*r3
                           h_num_tensor01,  //r3
                           &alpha, d_core12T231, CUDA_R_32F, h_num_tensor01, 
                           d_tensorT312, CUDA_R_32F, h_n_array[2],
                           &beta, d_ATBT3, CUDA_R_32F, h_num_core02,
                           CUDA_R_32F, algo);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    // cudaDeviceSynchronize();
	cudaFree(d_tensorT312);
	cudaFree(d_core12T231);
	// printMatrix_Device(h_num_core02,h_n_array[2],d_ATBT3,h_num_core02,"ATBT3");

	//求解线性方程  有问题 TODO解决
    cusolver_status = cusolverDnSgetrf_bufferSize(cusolverH,h_num_core02,h_num_core02,d_ATA3,h_num_core02,&lwork);
    assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
	cudaStat1 = cudaMalloc((void**)&d_work,sizeof(dt)*lwork);
	assert(cudaStat1 == cudaSuccess);
	cusolver_status = cusolverDnSgetrf(cusolverH,h_num_core02,h_num_core02,d_ATA3,h_num_core02,d_work,d_Ipiv,d_info);
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
	cusolver_status =cusolverDnSgetrs(cusolverH,CUBLAS_OP_N,h_num_core02,h_n_array[2],d_ATA3,h_num_core02,d_Ipiv,d_ATBT3,h_num_core02,d_info);
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
	cudaStat1 = cudaDeviceSynchronize();
	assert(cudaStat1 == cudaSuccess);
	cudaFree(d_ATA3);
	// printMatrix_Device(h_num_core02, h_n_array[2], d_ATBT3, h_num_core02, "ATBT3T");

    //获得core3
    cublas_status = cublasSgeam(cublasH,CUBLAS_OP_T,CUBLAS_OP_N,h_n_array[2],h_num_core02,&alpha,d_ATBT3,h_num_core02,&beta,d_core3T213,h_n_array[2],d_core3T213,h_n_array[2]);
	assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    cudaFree(d_ATBT3);
 //    printMatrix_Device(h_n_array[2], h_num_core02, d_core3T213, h_n_array[2], "core3T213Matrix");
	// printTensor(d_core3T213, h_n_array[2], h_r_array[2], h_r_array[0], "core3T213");
	dim3 block_core3T213((h_num_core3+512-1)/512,1,1); //for core1T213
	tensorToMode2<<<block_core3T213,threads>>>(d_core3T213, d_core3, h_n_array[2], h_r_array[2], h_r_array[0]);
	// printTensor(d_core3, h_r_array[2], h_n_array[2] ,h_r_array[0], "core3");
	cudaFree(d_core3T213);

	if(i == maxiter-1){
		// recover to X3' which is same to X
		// X3'= (BkrA)*C' 
		// printTensor(d_tensor, h_n_array[0], h_n_array[1], h_n_array[2], "tensor");
		dt *d_rec = NULL; float sh = 0.0, xia = 0.0;
		cudaStat1 = cudaMalloc((void**)&d_rec,sizeof(float)*h_num_tensor);
		assert(cudaStat1 == cudaSuccess);
		cublas_status = cublasGemmEx(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                           h_r_array[0]*h_num_tensor01, //r1*r3
                           h_n_array[2]*h_r_array[0], //r1*r3
                           h_r_array[2], //n1*n2
                           &alpha, d_core12, CUDA_R_32F, h_r_array[0]*h_num_tensor01, 
                           d_core3, CUDA_R_32F, h_r_array[2],
                           &beta, d_rec, CUDA_R_32F, h_r_array[0]*h_num_tensor01,
                           CUDA_R_32F, algo);
   	 	assert(cublas_status == CUBLAS_STATUS_SUCCESS);
		cudaFree(d_core12);
	///TODO  r0 != 1 的情况  参照matlab
		// printTensor(d_rec, h_n_array[0], h_n_array[1], h_n_array[2], "recover");
	
 		dim3 block_tensor((h_num_tensor+512-1)/512,1,1); //for core1T213
		elemin<<<block_tensor,threads>>>(d_tensor,d_rec,h_num_tensor);
		//cout<<"error "<<endl; printTensor(d_rec,a*b,c,1);
		//error rate = norm(res)/norm(X);

		cublasSnrm2(cublasH,h_num_tensor,d_rec,1,&sh);
		//cout<<"shang "<<endl; cout<<sh<<endl;
		cublasSnrm2(cublasH,h_num_tensor,d_tensor,1,&xia);
		cudaDeviceSynchronize();
		//cout<<"xia "<<endl; cout<<xia<<endl;
		cout<<"error rate "<<sh/xia<<endl;
		cudaFree(d_rec);
	}
}


/// 4. 在最后一次迭代计算误差
    
    
    // cudaFree(d_core23T231);
    cudaFree(d_work);
    cudaFree(d_info);
  	cudaFree(d_Ipiv);
    cudaFree(d_tensor);
    cudaFree(d_core1);
    cudaFree(d_core2);
    cudaFree(d_core3);
    cusolverDnDestroy(cusolverH);
    cublasDestroy(cublasH);
}