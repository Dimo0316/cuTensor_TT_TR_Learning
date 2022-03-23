#include "head.h"

__global__ void tranforArrayToDiagonalMatrix(float* array, float* diagMatrix, int n){
    long long i = blockIdx.x * blockDim.x + threadIdx.x;
    const long long temp = blockDim.x*gridDim.x;
    while(i < n*n){
        long row = i/n;
        long col = i%n;
        if(row == col)
            diagMatrix[i] = array[row];
        else
            diagMatrix[i] = 0;
        i+=temp;
    }
    __syncthreads();
}

__global__ void hardm(dt *M,dt *N,dt *res,long  m){
     long i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<m){
        res[m-i] = M[i]*N[i];
    }
    __syncthreads();
}

__global__ void tensorToMode2(dt *T1,dt *T2,int m,int n,int k){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<m*n*k){
        int tube = i/(m*n);
        int row = (i-tube*(m*n))%m;
        int col = (i-tube*(m*n))/m;
        T2[tube*m*n+row*n+col] = T1[tube*m*n+col*m+row];
    }
    __syncthreads();
}

__global__ void tensorToMode3(dt *T1,dt *T2,int m,int n,int k){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<m*n*k){
        int tube = i/(m*n);
        int row = (i-tube*(m*n))%m;
        int col = (i-tube*(m*n))/m;
        T2[k*(col*m+row)+tube] = T1[tube*m*n+col*m+row];
    }
    __syncthreads();
}

float trdec_gpu_float_unopt(float* h_tensor, int* h_n_array, int* h_r_array, double* eps){
// /计算第一部分svd
    dt time0 = 0.0f;
    GPUTimer timer0;
    timer0.start();

    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status  = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    cudaError_t cudaStat5 = cudaSuccess;
    cudaError_t cudaStat6 = cudaSuccess;


    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    cublasSetMathMode(cublasH,CUBLAS_DEFAULT_MATH);

    dt *d_A = NULL;  /* device copy of A */
    dt *d_AT= NULL;
    dt *d_S = NULL;  /* singular values */
    dt *d_U = NULL;  /* left singular vectors */
    dt *d_VT = NULL;  /* right singular vectors */
    dt *d_G1 = NULL;
    dt *h_G1 = NULL;
    int *devInfo = NULL;
    dt *d_work = NULL;
    dt *d_rwork = NULL;

    int m = h_n_array[0];
    int n = h_n_array[1] * h_n_array[2];
    int lwork = 0;
    int info_gpu = 0;
    float norm_d_S = 0.0f;
    float truncated_Value = 0.0f;
    const dt alpha = 1.0;
    const float ep = eps[0]/1.414214;
    const dt beta = 0.0;

    cudaStat1 = cudaMalloc ((void**)&d_A   , sizeof(dt)*m*n);
    cudaStat2 = cudaMalloc ((void**)&d_AT  , sizeof(dt)*m*n);
    cudaStat3 = cudaMemcpy (d_AT, h_tensor, sizeof(dt)*m*n, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);

    // cout << " m1:" << m << "  n1:" << n << endl;
    
    //printMatrix_Device(m, n, d_AT, m, "A的转置前");
///判断是否要转置
    if(m < n){
        int temp = m;
        m = n;
        n = temp;  // (m > n)
        cublas_status = cublasSgeam(cublasH,
                                CUBLAS_OP_T, CUBLAS_OP_N, 
                                m, n,
                                &alpha,
                                d_AT, n,
                                &beta,
                                d_A, m,
                                d_A, m);
    }
    else{
        cublas_status = cublasSgeam(cublasH,
                                CUBLAS_OP_N, CUBLAS_OP_N, 
                                m, n,
                                &alpha,
                                d_AT, m,
                                &beta,
                                d_AT, n,
                                d_A, m);
    }
    assert(cublas_status == CUSOLVER_STATUS_SUCCESS);
    if (d_AT) cudaFree(d_AT);
    cudaDeviceSynchronize();

    int lda = m;  

/* step 3: copy A and B to device */
    
    cudaStat2 = cudaMalloc ((void**)&d_S   , sizeof(dt)*n);
    cudaStat3 = cudaMalloc ((void**)&d_U   , sizeof(dt)*m*n); //有问题
    cudaStat4 = cudaMalloc ((void**)&d_VT   , sizeof(dt)*n*n);
    cudaStat5 = cudaMalloc ((void**)&devInfo, sizeof(int));
    cudaStat1 = cudaMalloc ((void**)&d_G1  , sizeof(dt)*n*n);

    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);//有问题
    assert(cudaSuccess == cudaStat4);
    assert(cudaSuccess == cudaStat5);
    assert(cudaSuccess == cudaStat1);

    // cudaStat1 = cudaHostAlloc((void**)&h_S   , sizeof(dt)*n  , 0);
    
    // cudaStat3 = cudaHostAlloc((void**)&h_U   , sizeof(dt)*m*n, 0);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);

    // cout<< " m:" << m << " n:" << n  << " lda:" << lda <<endl;

// step 3: query working space of SVD
    cusolver_status = cusolverDnSgesvd_bufferSize(
        cusolverH,
        m,
        n,
        &lwork );
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    // cout << "part one work space: " <<sizeof(dt)*lwork << endl;
    cudaStat1 = cudaMalloc((void**)&d_work , sizeof(dt)*lwork);
    assert(cudaSuccess == cudaStat1);
    // printMatrix_Device(m, n, d_A, m, "A");
    
    /* step 5: compute SVD */
    signed char jobu = 'S'; // all m columns of U
    signed char jobvt = 'S'; // all n columns of VT
    // cout << "m:" << m <<" n:" <<n<< " lda:" <<lda<<endl;
    cusolver_status = cusolverDnSgesvd(
        cusolverH,
        jobu,
        jobvt,
        m,
        n,
        d_A,
        m,
        d_S,
        d_U,
        m,  // ldu
        d_VT,
        n, // ldvt,
        d_work,
        lwork,
        d_rwork,
        devInfo);
    cudaStat1 = cudaDeviceSynchronize();

    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    // cout << "m:" << m << " n:" << n <<" lda:" << lda <<endl;
    
    // printMatrix_Device(n, 1, d_S, n, "S");
    // printMatrix_Device(m, n, d_U, m, "U");
    // printMatrix_Device(n, n, d_VT, n, "VT");

    // printMatrix_Device(n, n, d_G1, n, "G1");
    // printMatrix_Device(n,1,d_S, n,"d_S");
    cudaStat4 = cudaMemcpyAsync(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat4);
    // printf("after gesvd: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);
   
    if (d_A) cudaFree(d_A);
    if (devInfo) cudaFree(devInfo);
    if (d_work ) cudaFree(d_work);
    if (d_rwork) cudaFree(d_rwork);

    dim3 threads1024(1024,1,1);
    dim3 blocks1024((n+1024-1)/1024,1,1);

//计算二范数。获得截断值
//     dt *d_Shardm = NULL;
//     cudaStat1 = cudaMalloc((void**)&d_Shardm, sizeof(dt)*n);
//     assert(cudaStat1 == cudaSuccess);
//     cublas_status = cublasSnrm2(cublasH, n,
//                 d_S, 1, &norm_d_S);
//     assert(cublas_status == CUBLAS_STATUS_SUCCESS);
//     cudaStat1 = cudaDeviceSynchronize();
//     assert(cudaStat1 == cudaSuccess);
//     truncated_Value = norm_d_S * ep * norm_d_S * ep;
// ///获得截断索引 truncat_number 也就是h_r_array
// ///TODO
//     float sumError = 0.0f;
//     int truncat_number_test = 0;
//     hardm<<<blocks1024,threads1024>>>(d_S,d_S,d_Shardm,n);
//     // printMatrix_Device(econ_s,1,d_S,econ_s,"S");
//     // printMatrix_Device(econ_s,1,d_Shardm,econ_s,"Shardm");
//     cout << "truncated_Value: " << truncated_Value << endl;
//     while(sumError < truncated_Value){
//         cublas_status = cublasSasum(cublasH, truncat_number_test,
//                             d_Shardm, 1, &sumError);
//         assert(cublas_status == CUBLAS_STATUS_SUCCESS);
//         truncat_number_test++;
//     }

//     if(d_Shardm) cudaFree(d_Shardm);   

    int truncat_number = h_r_array[0]*h_r_array[1];
    
    // cout << "truncat_number:" << truncat_number << endl;

///tr分解中获得tr里的   r1 和 r2  
	int r1 = h_r_array[0];
	int r2 = h_r_array[1];
	cout << "r1:" << r1 << " r2:" << r2 << endl;
	
///复制G1到内存 h_G1
    cublas_status = cublasSgeam(cublasH,
                                CUBLAS_OP_T, CUBLAS_OP_N, 
                                n, n,
                                &alpha,
                                d_VT, n,
                                &beta,
                                d_VT, n,
                                d_G1, n);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    cudaStat6 = cudaDeviceSynchronize();
    assert(cudaStat6 == cudaSuccess);
    if (d_VT   ) cudaFree(d_VT);
	

    float *d_Struncat = NULL;
///计算中间变量 第二部分
    dt *d_Smatrix = NULL;
    dt *d_AT2 = NULL, *d_G1truncat=NULL, *d_AT2Mid = NULL;
    cudaStat4 = cudaHostAlloc((void**)&h_G1  ,    sizeof(dt)*truncat_number*n, 0);
    cudaStat1 = cudaMalloc((void**)&d_Struncat,   sizeof(dt)*truncat_number);
    cudaStat5 = cudaMalloc((void**)&d_G1truncat,  sizeof(dt)*truncat_number*n);
    cudaStat2 = cudaMalloc((void**) &d_Smatrix,	  sizeof(dt)*truncat_number*truncat_number);
    cudaStat3 = cudaMalloc((void**) &d_AT2,       sizeof(dt)*truncat_number*m);
    cudaStat3 = cudaMalloc((void**) &d_AT2Mid,    sizeof(dt)*truncat_number*m);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);
    assert(cudaStat4 == cudaSuccess);

///TODO 将G1 从n1*r1*r2 转变成 r1*n1*r2

    dim3 threadsnr1r2(1024,1,1);
    dim3 blocksnr1r2((r1*r2*m+1024-1)/1024,1,1);
    
	cublas_status = cublasScopy(cublasH, truncat_number*n,
                            d_G1, 1,
                            d_G1truncat, 1);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaStat1 == cudaSuccess);


	tensorToMode2<<<blocksnr1r2,threadsnr1r2>>>(d_G1truncat, d_G1, n, r1, r2);
	cudaStat6 = cudaDeviceSynchronize();
	assert(cudaStat6 == cudaSuccess);
    cudaStat5 = cudaMemcpyAsync(h_G1, d_G1, sizeof(dt)*truncat_number*n, cudaMemcpyDeviceToHost);
    cudaStat6 = cudaDeviceSynchronize();
    assert(cudaStat5 == cudaSuccess);
    assert(cudaStat6 == cudaSuccess);

    cublas_status = cublasScopy(cublasH, truncat_number,
                            d_S, 1,
                            d_Struncat, 1);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaStat1 == cudaSuccess);
    if(d_S) cudaFree(d_S);
    if(d_G1) cudaFree(d_G1);
    if(d_G1truncat) cudaFree(d_G1truncat);
    
    // printMatrix_Device(n,n,d_Smatrix,n,"Smatrix1");
    dim3 thread(512,1,1);
    dim3 blocks((n*n+1024-1)/1024,1,1);

    tranforArrayToDiagonalMatrix<<<blocks,thread>>>(d_Struncat,d_Smatrix,truncat_number);
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaStat1 == cudaSuccess);
    // printMatrix_Device(n,n,d_Smatrix,n,"Smatrix");
    cublas_status = cublasGemmEx(cublasH, 
        CUBLAS_OP_N, CUBLAS_OP_T,
        truncat_number, m, truncat_number,
        &alpha,
        d_Smatrix, CUDA_R_32F, truncat_number,
        d_U, CUDA_R_32F, m,
        &beta,
        d_AT2Mid, CUDA_R_32F, truncat_number,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaStat1 == cudaSuccess);

    dim3 threads512(1024,1,1);
    dim3 blocks512((r1*r2*m+1024-1)/1024,1,1);
    // GPUTimer timer2;
    // timer2.start();
///TODO 将d_AT2 从r1*r2*m 转变成 r2*m*r1
	tensorToMode2<<<blocks512,threads512>>>(d_AT2Mid,d_AT2,r1,r2,m);//r1 r2 m
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaStat1 == cudaSuccess);
    tensorToMode3<<<blocks512,threads512>>>(d_AT2,d_AT2Mid,r2,r1,m); //r2 r1 m
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaStat1 == cudaSuccess);
    tensorToMode2<<<blocks512,threads512>>>(d_AT2Mid,d_AT2,m,r2,r1); // m r2 r1
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaStat1 == cudaSuccess);
    // float time = timer2.seconds();
    // cout << "----------------------------------\nchange gemm run time: " << time << endl;
    // printMatrix_Device(n,m,d_AT2,n,"AT2");
    
    if(d_Smatrix) cudaFree(d_Smatrix);
    // if(d_S) cudaFree(d_S);
    if(d_U) cudaFree(d_U);
    if(d_AT2Mid) cudaFree(d_AT2Mid);


///TODO 改变 计算的m n数值
///计算第三部分svd
    cusolverH = NULL;
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    dt *d_S2 = NULL;  /* singular values */
    dt *d_U2 = NULL;  /* left singular vectors */
    dt *d_VT2 = NULL;  /* right singular vectors */
    dt *d_G3 = NULL;

    dt *h_G2 = NULL;
    dt *h_G3 = NULL;
    int *devInfo2 = NULL;
    dt *d_work2 = NULL;
    dt *d_rwork2 = NULL;
    
    assert(cudaStat1 == cudaSuccess);
    m = h_n_array[1]*r2;
    n = h_n_array[2]*r1;
    // cout << "m:" << m << " n:" << n << endl;
   
    cudaStat1 =  cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);
    // printMatrix_Device(m,n,d_A2,m,"A2");
    lda = m;  

/* step 3: copy A and B to device */
    cudaStat1 = cudaMalloc ((void**)&d_G3   , sizeof(dt)*n*n);
    cudaStat2 = cudaMalloc ((void**)&d_S2   , sizeof(dt)*n);
    cudaStat3 = cudaMalloc ((void**)&d_U2   , sizeof(dt)*m*n);
    cudaStat4 = cudaMalloc ((void**)&d_VT2   , sizeof(dt)*n*n);
    cudaStat5 = cudaMalloc ((void**)&devInfo2, sizeof(int));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);
    assert(cudaSuccess == cudaStat5);

    int lwork2 = 0;
// step 3: query working space of SVD
    cusolver_status = cusolverDnSgesvd_bufferSize(
        cusolverH,
        m,
        n,
        &lwork2 );
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    // cout << "part three work space: " << sizeof(dt)*lwork2 << endl;
    cudaStat1 = cudaMalloc((void**)&d_work2 , sizeof(dt)*lwork2);
    assert(cudaSuccess == cudaStat1);
     // cout << "m:" << m << " n:" << n <<" lda:" << lda <<endl;
    
    cusolver_status = cusolverDnSgesvd(
        cusolverH,
        jobu,
        jobvt,
        m,
        n,
        d_AT2,
        lda,
        d_S2,
        d_U2,
        m,  // ldu
        d_VT2,
        n, // ldvt,
        d_work2,
        lwork2,
        d_rwork2,
        devInfo2);
    cudaStat1 = cudaDeviceSynchronize();
   
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);
    if(d_AT2) cudaFree(d_AT2);
    
    // printMatrix_Device(n, 1, d_S2, n, "S2");
    // printMatrix_Device(m, n, d_U2, m, "U2");
    // printMatrix_Device(n, n, d_VT2, n, "VT2");
    
    
    //计算获得G3 TODO改成非优化版本
    // cublas_status = cublasSdgmm(cublasH,
    //                             CUBLAS_SIDE_LEFT,
    //                             n, n,
    //                             d_V2, n,
    //                             d_S2, 1,
    //                             d_G3, n);
    // assert(statusCublas == CUBLAS_STATUS_SUCCESS);
    // cudaStat1 = cudaDeviceSynchronize();
    // assert(cudaSuccess == cudaStat1);

//     //计算二范数。获得截断值
//     dt *d_Shardm2 = NULL;
//     cudaStat1 = cudaMalloc((void**)&d_Shardm2, sizeof(dt)*n);
//     assert(cudaStat1 == cudaSuccess);
//     cublas_status = cublasSnrm2(cublasH, n,
//                 d_S2, 1, &norm_d_S);
//     assert(cublas_status == CUBLAS_STATUS_SUCCESS);
//     cudaStat1 = cudaDeviceSynchronize();
//     assert(cudaStat1 == cudaSuccess);
//     truncated_Value = norm_d_S * ep * norm_d_S * ep;
//      // cout <<"truncated :" << truncated  << " norm_d_S:" << norm_d_S << endl;
// ///获得截断索引 truncat_number 也就是h_r_array
// ///TODO
//     sumError = 0.0f;
//     truncat_number_test = 0;
//     dim3 blocks10242((n+1024-1)/1024,1,1);
//     hardm<<<blocks10242,threads1024>>>(d_S2,d_S2,d_Shardm2,n);
//     // printMatrix_Device(n,1,d_S2,n,"S2");
//     // printMatrix_Device(econ_s,1,d_Shardm,econ_s,"Shardm");
//     cout << "truncated_Value2: " << truncated_Value << endl;
//     while(sumError < truncated_Value){
//         cublas_status = cublasSasum(cublasH, truncat_number_test,
//                             d_Shardm2, 1, &sumError);
//         assert(cublas_status == CUBLAS_STATUS_SUCCESS);
//         // cout << "sumError2" << sumError << endl;
//         truncat_number_test++;
//     }
//     // time = timer2.seconds();
//     // cout << "truncat_number_test2:" << econ_s2 - truncat_number_test + 3 <<endl;
//     if(d_Shardm2) cudaFree(d_Shardm2);

///获得截断索引 truncat_number 也就是h_r_array
///TODO

    truncat_number = h_r_array[2];
    cout << "truncat_number2: " << truncat_number << endl;
    
    dt *d_Smatrix2 = NULL, *d_Struncat2 = NULL;//, *d_Vtruncat2 = NULL;
    cudaStat1 = cudaMalloc((void**)&d_Struncat2, sizeof(dt)*truncat_number);
    cudaStat2 = cudaMalloc((void**) &d_Smatrix2, sizeof(dt)*truncat_number*truncat_number);
    cudaStat3 = cudaHostAlloc((void**)&h_G2  , sizeof(dt)*m*truncat_number, 0);
    cudaStat4 = cudaHostAlloc((void**)&h_G3  , sizeof(dt)*n*truncat_number, 0);
    cudaStat5 = cudaDeviceSynchronize();
    assert(cudaStat1 == cudaSuccess);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaStat4 == cudaSuccess);
    assert(cudaStat5 == cudaSuccess);


    ///截断S2
    cublas_status = cublasScopy(cublasH, truncat_number,
                            d_S2, 1,
                            d_Struncat2, 1);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaStat1 == cudaSuccess);
    if(d_S2) cudaFree(d_S2);

    // GPUTimer timer4; 
    // timer4.start();
    dim3 thread2(512,1,1);
    dim3 blocks2((n*n+1024-1)/1024,1,1);
    tranforArrayToDiagonalMatrix<<<blocks2,thread2>>>(d_Struncat2,d_Smatrix2,truncat_number);
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaStat1 == cudaSuccess);
    if(d_Struncat2) cudaFree(d_Struncat2); d_Struncat2=NULL;
    cublas_status = cublasGemmEx(cublasH, 
        CUBLAS_OP_N, CUBLAS_OP_N,
        truncat_number, n, truncat_number,
        &alpha,
        d_Smatrix2, CUDA_R_32F, truncat_number,
        d_VT2, CUDA_R_32F, truncat_number,
        &beta,
        d_G3, CUDA_R_32F, truncat_number,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaStat1 == cudaSuccess);

    // printMatrix_Device(n,n,d_G3,n,"G3");

    cudaStat4 = cudaMemcpyAsync(&info_gpu, devInfo2, sizeof(int), cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpyAsync(h_G2, d_U2, sizeof(dt)*m*truncat_number, cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpyAsync(h_G3, d_G3, sizeof(dt)*n*truncat_number, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat4);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat2);

    // printf("after gesvd: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);
  
    // if (d_S2) cudaFree(d_S2);
    if (d_U2) cudaFree(d_U2);
    if (d_VT2) cudaFree(d_VT2);
    if (d_G3) cudaFree(d_G3);
    if (devInfo2) cudaFree(devInfo2);
    if (d_work2 ) cudaFree(d_work2);
    if (d_rwork2) cudaFree(d_rwork2);
    if (d_Smatrix2) cudaFree(d_Smatrix2);
    
    if (h_G1) cudaFreeHost(h_G1);
    if (h_G2) cudaFreeHost(h_G2);
    if (h_G3) cudaFreeHost(h_G3);
    if (cublasH ) cublasDestroy(cublasH);
    if (cusolverH) cusolverDnDestroy(cusolverH);

    time0 = timer0.seconds();
    return time0;
}