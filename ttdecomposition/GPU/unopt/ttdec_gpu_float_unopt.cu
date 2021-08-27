#include "head.h"

void ttdec_gpu_float_unopt(float* h_tensor, int* h_n_array, double eps){
///计算第一部分svd
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;

    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status  = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    cudaError_t cudaStat5 = cudaSuccess;


    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);

    dt *d_A = NULL;  /* device copy of A */
    dt *d_AT= NULL;
    dt *d_S = NULL;  /* singular values */
    dt *h_S = NULL;
    dt *d_U = NULL;  /* left singular vectors */
    dt *d_VT = NULL;  /* right singular vectors */
    dt *d_G1 = NULL;
    dt *h_G1 = NULL;
    dt *h_U = NULL;
    int *devInfo = NULL;
    dt *d_work = NULL;
    dt *d_rwork = NULL;
    dt *d_W = NULL;  // W = S*VT

    int m = h_n_array[0];
    int n = h_n_array[1] * h_n_array[2];
    int lwork = 0;
    int info_gpu = 0;
    // const dt h_one = 1;
    // const dt h_minus_one = -1;
    const dt alpha = 1.0;
    const dt beta = 0.0;

    cudaStat1 = cudaMalloc ((void**)&d_A   , sizeof(dt)*m*n);
    cudaStat2 = cudaMalloc ((void**)&d_AT  , sizeof(dt)*m*n);
    cudaStat3 = cudaMemcpy (d_AT, h_tensor, sizeof(dt)*m*n, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    //cout << " m1:" << m << "  n1:" << n << endl;
    
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
    // printMatrix_Device(m, n, d_A, m, "A");

    int lda = m;  // 1500

/* step 3: copy A and B to device */
    
    cudaStat2 = cudaMalloc ((void**)&d_S   , sizeof(dt)*n);
    cudaStat3 = cudaMalloc ((void**)&d_U   , sizeof(dt)*m*m); //有问题
    cudaStat4 = cudaMalloc ((void**)&d_VT   , sizeof(dt)*n*n);
    cudaStat5 = cudaMalloc ((void**)&devInfo, sizeof(int));
    cudaStat1 = cudaMalloc ((void**)&d_W  , sizeof(dt)*lda*n);
    cudaStat1 = cudaMalloc ((void**)&d_G1  , sizeof(dt)*n*n);
    if(cudaStat3 == cudaErrorMemoryAllocation)
        cout << " cudaErrorMemoryAllocation "<<endl;
    if(cudaStat3 == cudaErrorInvalidValue)
        cout << " cudaErrorInvalidValue "<<endl;
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);//有问题
    assert(cudaSuccess == cudaStat4);
    assert(cudaSuccess == cudaStat5);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaHostAlloc((void**)&h_S   , sizeof(dt)*n  , 0);
    cudaStat2 = cudaHostAlloc((void**)&h_G1  , sizeof(dt)*n*n, 0);
    cudaStat3 = cudaHostAlloc((void**)&h_U   , sizeof(dt)*m*m, 0);
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
    cout << "part one work space: " <<sizeof(dt)*lwork << endl;
    cudaStat1 = cudaMalloc((void**)&d_work , sizeof(dt)*lwork);
    assert(cudaSuccess == cudaStat1);

    dt time = 0.0f;
    GPUTimer timer;
    timer.start();
    /* step 5: compute SVD */
    signed char jobu = 'A'; // all m columns of U
    signed char jobvt = 'A'; // all n columns of VT
    cusolver_status = cusolverDnSgesvd(
        cusolverH,
        jobu,
        jobvt,
        m,
        n,
        d_A,
        lda,
        d_S,
        d_U,
        lda,  // ldu
        d_VT,
        n, // ldvt,
        d_work,
        lwork,
        d_rwork,
        devInfo);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    time = timer.seconds();
    cout << "----------------------------------\npart one svd run time: " << time << endl;
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

    // printMatrix_Device(n,1,d_S, n,"d_S");

    cudaStat4 = cudaMemcpyAsync(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    cudaStat5 = cudaMemcpyAsync(h_S, d_S, sizeof(dt)*n, cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpyAsync(h_G1, d_G1, sizeof(dt)*n*n, cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpyAsync(h_U, d_U, sizeof(dt)*m*m, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat4);
    assert(cudaSuccess == cudaStat5);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat2);

    printf("after gesvd: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);
    // printf("=====\n");

//     // step 6: |A - U*S*VT|
//     // W = S*VT
//     cublas_status = cublasSdgmm(
//         cublasH,
//         CUBLAS_SIDE_LEFT,
//         n,
//         n,
//         d_VT,
//         lda,
//         d_S,
//          1,
//         d_W,
//         lda);
//     assert(CUBLAS_STATUS_SUCCESS == cublas_status);

// // A := -U*W + A
//     cudaStat1 = cudaMemcpy(d_A, h_tensor, sizeof(dt)*lda*n, cudaMemcpyHostToDevice);
//     assert(cudaSuccess == cudaStat1);
//     cublas_status = cublasSgemm_v2(
//         cublasH,
//         CUBLAS_OP_N, // U
//         CUBLAS_OP_N, // W
//         m, // number of rows of A
//         n, // number of columns of A
//         n, // number of columns of U 
//         &h_minus_one, /* host pointer */
//         d_U, // U
//         lda,
//         d_W, // W
//         lda,
//         &h_one, /* hostpointer */
//         d_A,
//         lda);
//     assert(CUBLAS_STATUS_SUCCESS == cublas_status);

//     dt dR_fro = 0.0;
//     cublas_status = cublasSnrm2_v2(
//         cublasH, lda*n, d_A, 1, &dR_fro);
//     assert(CUBLAS_STATUS_SUCCESS == cublas_status);

//     printf("|A - U*S*VT| = %E \n", dR_fro);
    if (d_G1) cudaFree(d_G1);
    if (d_A) cudaFree(d_A);
    if (d_S) cudaFree(d_S);
    if (d_U) cudaFree(d_U);
    if (d_VT   ) cudaFree(d_VT);
    if (devInfo) cudaFree(devInfo);
    if (d_work ) cudaFree(d_work);
    if (d_rwork) cudaFree(d_rwork);
    //if (d_W    ) cudaFree(d_W);

    
///计算中间变量 第二部分
    dt *h_Smatrix = NULL;
    dt *d_Smatrix = NULL;
    dt *h_Umatrix = NULL;
    dt *d_Umatrix = NULL;
    dt *d_AT2 = NULL;
    
    cudaStat1 = cudaHostAlloc((void**)&h_Smatrix, sizeof(dt)*n*n, 0);
    cudaStat2 = cudaHostAlloc((void**)&h_Umatrix, sizeof(dt)*m*n, 0);
    cudaStat3 = cudaMalloc((void**) &d_Smatrix,    sizeof(dt)*n*n);
    cudaStat4 = cudaMalloc((void**) &d_Umatrix,    sizeof(dt)*m*n);
    cudaStat5 = cudaMalloc((void**) &d_AT2,         sizeof(dt)*n*m);
    cudaDeviceSynchronize();
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);
    assert(cudaStat4 == cudaSuccess);
    assert(cudaStat5 == cudaSuccess);

    //初始化h_Smatrix
    for(int i = 0; i < n*n; ++i){
        h_Smatrix[i] = 0;
    }
  
    for(int i = 0; i < n; ++i){
        h_Smatrix[n*i+i] = h_S[i];
    }

    //裁剪u矩阵
    for(int i = 0; i < m*n; ++i){
        h_Umatrix[i] = h_U[i];
    }

    cudaStat1 = cudaMemcpyAsync(d_Umatrix, h_Umatrix, sizeof(dt)*m*n, cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpyAsync(d_Smatrix, h_Smatrix, sizeof(dt)*n*n, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);

    GPUTimer timer2;
    timer2.start();
    cublasStatus_t  statusCublas = cublasGemmEx(cublasH, 
        CUBLAS_OP_N, CUBLAS_OP_T,
        n, m, n,
        &alpha,
        d_Smatrix, CUDA_R_32F, n,
        d_Umatrix, CUDA_R_32F, m,
        &beta,
        d_AT2, CUDA_R_32F, n,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    assert(statusCublas == CUBLAS_STATUS_SUCCESS);
    cudaDeviceSynchronize();
    time = timer2.seconds();
    cout << "----------------------------------\n part two gemm run time: " << time << endl;

    if(h_U) cudaFreeHost(h_U);
    if(h_S) cudaFreeHost(h_S);
    if(h_Umatrix) cudaFreeHost(h_Umatrix);
    if(h_Smatrix) cudaFreeHost(h_Smatrix);
    if(d_Smatrix) cudaFree(d_Smatrix);
    if(d_Umatrix) cudaFree(d_Umatrix);

///计算第三部分svd

    dt *d_A2= NULL;
    dt *d_S2 = NULL;  /* singular values */
    dt *d_U2 = NULL;  /* left singular vectors */
    dt *d_VT2 = NULL;  /* right singular vectors */
    dt *h_G2T = NULL;
    dt *h_G2 = NULL;
    dt *h_G3 = NULL;
    int *devInfo2 = NULL;
    dt *d_work2 = NULL;
    dt *d_rwork2 = NULL;
    
    cudaStat1 = cudaMalloc ((void**)&d_A2, sizeof(dt)*m*n);
    assert(cudaStat1 == cudaSuccess);
    m = h_n_array[0]*h_n_array[1];
    n = h_n_array[2];

    if(m < n){
        int temp = m;
        m = n;
        n = temp;  // (m > n)
        cublas_status = cublasSgeam(cublasH,
                                CUBLAS_OP_T, CUBLAS_OP_N, 
                                m, n,
                                &alpha,
                                d_AT2, n,
                                &beta,
                                d_A2, m,
                                d_A2, m);
    }
    else{
        cublas_status = cublasSgeam(cublasH,
                                CUBLAS_OP_N, CUBLAS_OP_N, 
                                m, n,
                                &alpha,
                                d_AT2, m,
                                &beta,
                                d_A2, m,
                                d_A2, m);
    }
    assert(cublas_status == CUSOLVER_STATUS_SUCCESS);

    if (d_AT2) cudaFree(d_AT2);
    cudaDeviceSynchronize();
    // printMatrix_Device(m,n,d_A2,m,"A2");
    lda = m;  

/* step 3: copy A and B to device */
    
    cudaStat2 = cudaMalloc ((void**)&d_S2   , sizeof(dt)*n);
    cudaStat3 = cudaMalloc ((void**)&d_U2   , sizeof(dt)*m*m);
    cudaStat4 = cudaMalloc ((void**)&d_VT2   , sizeof(dt)*n*n);
    cudaStat5 = cudaMalloc ((void**)&devInfo2, sizeof(int));
   
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);
    assert(cudaSuccess == cudaStat5);
    assert(cudaSuccess == cudaStat1);

    cudaStat2 = cudaHostAlloc((void**)&h_G2T , sizeof(dt)*m*m, 0);
    cudaStat3 = cudaHostAlloc((void**)&h_G2  , sizeof(dt)*m*n, 0);
    cudaStat4 = cudaHostAlloc((void**)&h_G3  , sizeof(dt)*n*n, 0);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);

// step 3: query working space of SVD
    cusolver_status = cusolverDnSgesvd_bufferSize(
        cusolverH,
        m,
        n,
        &lwork );
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    cout << "part three work space: " <<sizeof(dt)*lwork << endl;
    cudaStat1 = cudaMalloc((void**)&d_work2 , sizeof(dt)*lwork);
    assert(cudaSuccess == cudaStat1);

    time = 0.0f;
    GPUTimer timer3;
    timer3.start();
    /* step 5: compute SVD */
    // signed char jobu = 'A'; // all m columns of U
    // signed char jobvt = 'A'; // all n columns of VT
    cusolver_status = cusolverDnSgesvd(
        cusolverH,
        jobu,
        jobvt,
        m,
        n,
        d_A2,
        lda,
        d_S2,
        d_U2,
        lda,  // ldu
        d_VT2,
        n, // ldvt,
        d_work2,
        lwork,
        d_rwork2,
        devInfo2);
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    assert(cudaSuccess == cudaStat1);

    time = timer3.seconds();
    cout << "----------------------------------\npart three svd run time: " << time << endl;

    // printMatrix_Device(m,m,d_U2, m,"d_U2");
    // printMatrix_Device(n,1,d_S2, n,"d_S2");
    // printMatrix_Device(n,n,d_VT2, n,"d_VT2");

    cudaStat4 = cudaMemcpyAsync(&info_gpu, devInfo2, sizeof(int), cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpyAsync(h_G2T, d_U2, sizeof(dt)*m*m, cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpyAsync(h_G3, d_VT2, sizeof(dt)*n*n, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat4);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat2);

    printf("after gesvd: info_gpu = %d\n", info_gpu);
    assert(0 == info_gpu);
    // printf("=====\n");

    for(int i = 0; i < m*n; ++i){
        h_G2[i] = h_G2T[i];
    }

    if (d_A2) cudaFree(d_A2);
    if (d_S2) cudaFree(d_S2);
    if (d_U2) cudaFree(d_U2);
    if (d_VT2) cudaFree(d_VT2);
    if (devInfo2) cudaFree(devInfo2);
    if (d_work2 ) cudaFree(d_work2);
    if (d_rwork2) cudaFree(d_rwork2);

    //printf("step2 - calculate mid success--------2*12--------- \n");
    // if(d_AT2) cudaFreeHost(d_AT2);
    if (h_G2T) cudaFreeHost(h_G2T);
    if (h_G1) cudaFreeHost(h_G1);
    if (h_G2) cudaFreeHost(h_G2);
    if (h_G3) cudaFreeHost(h_G3);
    if (cublasH ) cublasDestroy(cublasH);
    if (cusolverH) cusolverDnDestroy(cusolverH);
    // cudaDeviceReset();
    // printf("====finish====");
}