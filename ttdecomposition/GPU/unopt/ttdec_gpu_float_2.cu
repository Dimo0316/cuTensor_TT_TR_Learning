#include "head.h"

__global__ void hardm(dt *M,dt *N,dt *res,long  m){
     long i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<m){
        res[m-i] = M[i]*N[i];
    }
    __syncthreads();
}

void ttdec_gpu_float(float* h_tensor, int* h_n_array, int* h_r_array,double eps){
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    gesvdjInfo_t gesvdj_params = NULL;
    
    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    cudaError_t cudaStat5 = cudaSuccess;

    dt *d_A = NULL;  /* device copy of A */
    dt *d_S = NULL;  /* singular values */
    dt *d_U = NULL;  /* left singular vectors */
    dt *d_V = NULL;  /* right singular vectors */
    dt *h_G1 = NULL;
    //dt *h_S = NULL;
    int *d_info = NULL;  /* error info */
    int lwork = 0;       /* size of workspace */
    dt *d_work = NULL; /* devie workspace for gesvdj */
    int info = 0; 
    // const int m = h_n_array[work_time];
    // const int n = h_n_array[1] * h_n_array[2] * h_n_array[3];
    const int m = h_n_array[0];  //1500
    const int n = h_n_array[1] * h_n_array[2];  //1500 * 1500 = 2250000
    
    const float ep = eps/1.414214;
    const int econ_s = min(m,n); //1500
    const int lda = m;  // 1500
    const int ldu = lda;  // 1500
    const int ldv = n;  // 2250000

    /* configuration of gesvdj  */
    const double tol = eps;
    const int max_sweeps = 100;
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
    const int econ = 1;
   
    /* numerical results of gesvdj  */
    double residual = 0;
    int executed_sweeps = 0;
    float norm_d_S = 0.0f;
    float truncated_Value = 0.0f;
    // printf("step1 ------------ \n");
    // printf("tol = %E, default value is machine zero \n", tol);
    // printf("max. sweeps = %d, default value is 100\n", max_sweeps);
    // printf("econ = %d \n", econ);

    // GPUTimer timer0;
    // timer0.start();
    /* step 1: create cusolver handle, bind a stream */
    status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cudaStat1);

    status = cusolverDnSetStream(cusolverH, stream);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* step 2: configuration of gesvdj */
    status = cusolverDnCreateGesvdjInfo(&gesvdj_params);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* default value of tolerance is machine zero */
    status = cusolverDnXgesvdjSetTolerance(
        gesvdj_params,
        tol);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* default value of max. sweeps is 100 */
    status = cusolverDnXgesvdjSetMaxSweeps(
        gesvdj_params,
        max_sweeps);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* step 3: copy A and B to device */
    cudaStat1 = cudaMalloc ((void**)&d_A   , sizeof(dt)*lda*n);
    cudaStat2 = cudaMalloc ((void**)&d_S   , sizeof(dt)*econ_s);
    cudaStat3 = cudaMalloc ((void**)&d_U   , sizeof(dt)*ldu*econ_s);
    cudaStat4 = cudaMalloc ((void**)&d_V   , sizeof(dt)*ldv*econ_s);//
    cudaStat5 = cudaMalloc ((void**)&d_info, sizeof(int));
    //cudaHostAlloc((void**) &h_S, sizeof(dt)*econ_s,0);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);
    assert(cudaSuccess == cudaStat5);
    
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(d_A, h_tensor, sizeof(dt)*lda*n, cudaMemcpyHostToDevice);
    //cudaStat1 = cudaMemcpyAsync(d_A, h_tensor, sizeof(dt)*lda*n, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    // cout<< " m:" << m << " n:" << n  <<endl;
    // cout << " lda:" << lda << " ldu:" << ldu << " ldv:" << ldv << " eps:" << eps <<endl;

/* step 4: query workspace of SVD */
    status = cusolverDnSgesvdj_bufferSize(
        cusolverH,
        jobz, /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
              /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
        econ, /* econ = 1 for economy size */
        m,    /* nubmer of rows of A, 0 <= m */
        n,    /* number of columns of A, 0 <= n  */
        d_A,  /* m-by-n */
        lda,  /* leading dimension of A */
        d_S,  /* min(m,n) */
              /* the singular values in descending order */
        d_U,  /* m-by-m if econ = 0 */
              /* m-by-min(m,n) if econ = 1 */
        ldu,  /* leading dimension of U, ldu >= max(1,m) */
        d_V,  /* n-by-n if econ = 0  */
              /* n-by-min(m,n) if econ = 1  */
        ldv,  /* leading dimension of V, ldv >= max(1,n) */
        &lwork,
        gesvdj_params);
    // if( CUSOLVER_STATUS_NOT_INITIALIZED == status)
    //     cout << "CUSOLVER_STATUS_NOT_INITIALIZED" <<endl;

    // if( CUSOLVER_STATUS_INVALID_VALUE == status)
    //     cout << "CUSOLVER_STATUS_INVALID_VALUE  " <<endl;

    // if( CUSOLVER_STATUS_INTERNAL_ERROR == status)
    //     cout << "CUSOLVER_STATUS_INTERNAL_ERROR" <<endl; 

    assert(CUSOLVER_STATUS_SUCCESS == status);
    // cout << "-------------------------\npart one work space: " <<sizeof(dt)*lwork << endl;
    cudaStat1 = cudaMalloc((void**)&d_work , sizeof(dt)*lwork);
    assert(cudaSuccess == cudaStat1);

    // printMatrix_Device(lda,n,d_A,lda,"A");
    // printMatrix(lda,n,h_tensor,lda,"hA");

    // dt time = 0.0f;
    // GPUTimer timer;
    // timer.start();
    /* step 5: compute SVD */
    status = cusolverDnSgesvdj(
        cusolverH,
        jobz,  /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
               /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
        econ,  /* econ = 1 for economy size */
        m,     /* nubmer of rows of A, 0 <= m */
        n,     /* number of columns of A, 0 <= n  */
        d_A,   /* m-by-n */
        lda,   /* leading dimension of A */
        d_S,   /* min(m,n)  */
               /* the singular values in descending order */
        d_U,   /* m-by-m if econ = 0 */
               /* m-by-min(m,n) if econ = 1 */
        ldu,   /* leading dimension of U, ldu >= max(1,m) */
        d_V,   /* n-by-n if econ = 0  */
               /* n-by-min(m,n) if econ = 1  */
        ldv,   /* leading dimension of V, ldv >= max(1,n) */
        d_work,
        lwork,
        d_info,
        gesvdj_params);
    cudaStat1 = cudaDeviceSynchronize();
    // time = timer.seconds();
    // cout << "----------------------------------\n part one svd run time: " << time << endl;

    // printMatrix_Device(ldu,econ_s,d_U,ldu,"U");
    // printMatrix_Device(econ_s,1,d_S,econ_s,"S");
    // printMatrix_Device(ldv,econ_s,d_V,ldv,"V");

    assert(CUSOLVER_STATUS_SUCCESS == status);
    assert(cudaSuccess == cudaStat1);

    //cudaStat1 = cudaMemcpy(h_S, d_S, sizeof(dt)*econ_s, cudaMemcpyDeviceToHost);  
    cudaStat1 = cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    // cudaStat2 = cudaMemcpyAsync(h_G1, d_U, sizeof(dt)*ldu*econ_s, cudaMemcpyDeviceToHost);
    cudaStat3 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);


    if ( 0 == info ){
        printf("gesvdj converges \n");
    }else if ( 0 > info ){
        printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }else{
        printf("WARNING: info = %d : gesvdj does not converge \n", info );
    }
    

    status = cusolverDnXgesvdjGetSweeps(
        cusolverH,
        gesvdj_params,
        &executed_sweeps);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    status = cusolverDnXgesvdjGetResidual(
        cusolverH,
        gesvdj_params,
        &residual);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    printf("residual |A - U*S*V**H|_F = %E \n", residual );
    printf("number of executed sweeps = %d \n", executed_sweeps );

    // if (d_A) cudaFree(d_A);
    // if (d_U) cudaFree(d_U);
    
    if (d_info) cudaFree(d_info);
    if (d_work ) cudaFree(d_work);

    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cublasHandle_t handle = NULL;
    cublasCreate(&handle);

//计算二范数。获得截断值
    dt *d_Shardm = NULL;
    cudaStat1 = cudaMalloc((void**)&d_Shardm, sizeof(dt)*econ_s);
    assert(cudaStat1 == cudaSuccess);
    cublas_status = cublasSnrm2(handle, econ_s,
                d_S, 1, &norm_d_S);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaStat1 == cudaSuccess);
    truncated_Value = norm_d_S * ep * norm_d_S * ep;
///获得截断索引 truncat_number 也就是h_r_array
///TODO
    // printMatrix_Device(econ_s,1,d_S,econ_s,"d_S");
    // cublas_status = cublasSdot (handle, econ_s,
    //                        d_S, econ_s,
    //                        d_S, econ_s,
    //                        d_Sdot);
    // assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    // cudaStat1 = cudaDeviceSynchronize();
    // assert(cudaStat1 == cudaSuccess);
    // printMatrix_Device(econ_s,1,d_Sdot,econ_s,"d_Sdot");
    // float time = 0.0f;
    // GPUTimer timer;
    // timer.start();
    float sumError = 0.0f;
    int truncat_number_test = 0;
    dim3 threads(1024,1,1);
    dim3 blocks((econ_s+1024-1)/1024,1,1);
    hardm<<<blocks,threads>>>(d_S,d_S,d_Shardm,econ_s);
    // printMatrix_Device(econ_s,1,d_S,econ_s,"S");
    // printMatrix_Device(econ_s,1,d_Shardm,econ_s,"Shardm");
    cout << "truncated_Value" << truncated_Value << endl;
    while(sumError < truncated_Value){
        cublas_status = cublasSasum(handle, truncat_number_test,
                            d_Shardm, 1, &sumError);
        assert(cublas_status == CUBLAS_STATUS_SUCCESS);
        // cout << "sumError" << sumError << endl;
        truncat_number_test++;
    }
    // time = timer.seconds();
    // cout << "truncat_number_test:" << econ_s - truncat_number_test + 3 <<endl;
    if(d_Shardm) cudaFree(d_Shardm);
    h_r_array[1] = econ_s - truncat_number_test + 2;
    // h_r_array[1] = min(econ_s - truncat_number_test + 2, h_r_array[1]);
    int truncat_number = h_r_array[1];
    cout << "truncat_number:" << truncat_number << endl;

///根绝截断值将核G1复制到内存
    cudaStat1 = cudaHostAlloc((void**)&h_G1, sizeof(dt)*ldu*truncat_number, 0);
    cudaStat2 = cudaMemcpyAsync(h_G1, d_U, sizeof(dt)*ldu*truncat_number, cudaMemcpyDeviceToHost);
    int comp_rat_number = truncat_number*ldu;
    // if (d_U) cudaFree(d_U);
///TODO 计算中间变量
    //TODO将s向量变成对角矩阵
    //TODO计算对角矩阵乘V转置矩阵
    //利用
    dt *d_A2 = NULL, *d_VT = NULL;//, *d_Struncat = NULL;//, *d_Vtruncat = NULL;
    cudaStat1 =cudaMalloc((void**)&d_A2, sizeof(dt)*truncat_number*n); //20*400
    cudaStat2 =cudaMalloc((void**)&d_VT, sizeof(dt)*econ_s*n); //20*400
    // cudaStat3 =cudaMalloc((void**)&d_Struncat, sizeof(dt)*truncat_number);
    // cudaStat4 =cudaMalloc((void**)&d_Vtruncat, sizeof(dt)*n*truncat_number);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);
    assert(cudaStat4 == cudaSuccess);
    cudaDeviceSynchronize();
    
  
    // cout <<"truncated :" << truncated  << " norm_d_S:" << norm_d_S << endl;
///截断S
    // cublas_status = cublasScopy(handle, truncat_number,
    //                         d_S, 1,
    //                         d_Struncat, 1);
    // assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    // cudaStat1 = cudaDeviceSynchronize();
    // assert(cudaStat1 == cudaSuccess);
    // if(d_S) cudaFree(d_S);
///截断V
    // printMatrix_Device(truncat_number, 1, d_Struncat, truncat_number, "Struncat");
    // printMatrix_Device(n, econ_s, d_V, n, "V");
    // cublas_status = cublasScopy(handle, n*truncat_number,
    //                         d_V, 1,
    //                         d_Vtruncat, 1);
    // assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    // cudaStat1 = cudaDeviceSynchronize();
    // assert(cudaStat1 == cudaSuccess);
    // if(d_V) cudaFree(d_V);
    // printMatrix_Device(n, truncat_number, d_Vtruncat, n, "Vtruncat");

    dt alpha = 1.0;
    dt beta = 0.0;
    //printMatrix_Device(n, econ_s, d_V, n, "V的设备");
        
    // GPUTimer timer2;
    // timer2.start();
//首先转置V n*econ_s 到 econ_s*n VT
    // cublas_status = cublasSgeam(handle,
    //                             CUBLAS_OP_T, CUBLAS_OP_N, 
    //                             truncat_number, n,
    //                             &alpha,
    //                             d_Vtruncat, n,
    //                             &beta,
    //                             d_Vtruncat, n,
    //                             d_VT, truncat_number);
    // assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    // cudaDeviceSynchronize();
    // if (d_Vtruncat) cudaFree(d_Vtruncat);
    cublas_status = cublasSgeam(handle,
                                CUBLAS_OP_T, CUBLAS_OP_N, 
                                econ_s, n,
                                &alpha,
                                d_V, n,
                                &beta,
                                d_V, n,
                                d_VT, econ_s);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    cudaDeviceSynchronize();
    if (d_V) cudaFree(d_V);
    //  printMatrix_Device( econ_s,n, d_VT, econ_s, "VT");
    // printMatrix_Device(truncat_number, n, d_VT, truncat_number, "VT的设备");
    // printMatrix_Device(econ_s, 1, d_S, econ_s, "S");
//再计算矩阵  列向量乘法  econ_s*econ_s  econ_s*n
    cublas_status = cublasSdgmm(handle,
                                CUBLAS_SIDE_LEFT,
                                truncat_number, n,
                                d_VT, econ_s,
                                d_S, 1,
                                d_A2, truncat_number);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    cudaDeviceSynchronize();

    // printMatrix_Device(truncat_number, n, d_VT, truncat_number, "VT2");
    // printMatrix_Device(truncat_number, 1, d_S, truncat_number, "S2");

    // time = timer2.seconds();
    // cout << "----------------------------------\npart two gemm run time: " << time << endl;

    if (d_VT) cudaFree(d_VT);
    if (d_S) cudaFree(d_S);
    // if (d_Struncat)  cudaFree(d_Struncat);

    // printMatrix_Device(truncat_number, n, d_A2, truncat_number, "A2");
    // printf("step2 - calculate mid success--------2*4--------- \n");

///TODO第二部分计算
    cusolverH = NULL;
    stream = NULL;
    gesvdj_params = NULL;

    dt *d_S2 = NULL;  /* singular values */
    dt *d_U2 = NULL;  /* left singular vectors */
    dt *d_V2 = NULL;  /* right singular vectors */
    dt *d_VT2 = NULL;
    dt *d_G3 = NULL;
    dt *h_G2 = NULL;
    dt *h_G3 = NULL;
    int *d_info2 = NULL;  /* error info */
    int lwork2 = 0;       /* size of workspace */
    float *d_work2 = NULL; /* devie workspace for gesvdj */
    int info2 = 0; 

    const int m2 = h_n_array[1] * h_r_array[1];  // 200*57
    const int n2 = h_n_array[2];  // 200
    const int econ_s2 = min(m2, n2);  // 4
    const int lda2 = m2;  // 6
    const int ldu2 = lda2;  // 6
    const int ldv2 = n2;  // 4
    //cout << " m2:" << m2 << "  n2:" << n2 << " econ_s2:" << econ_s2 << endl;

    /* numerical results of gesvdj  */
    residual = 0;
    executed_sweeps = 0;

    // printf("step3 ---------------------------- \n");
    // printf("tol = %E, default value is machine zero \n", tol);
    // printf("max. sweeps = %d, default value is 100\n", max_sweeps);
    // printf("econ = %d \n", econ);

 /* step 1: create cusolver handle, bind a stream */
    status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cudaStat1);

    status = cusolverDnSetStream(cusolverH, stream);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* step 2: configuration of gesvdj */
    status = cusolverDnCreateGesvdjInfo(&gesvdj_params);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* default value of tolerance is machine zero */
    status = cusolverDnXgesvdjSetTolerance(
        gesvdj_params,
        tol);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* default value of max. sweeps is 100 */
    status = cusolverDnXgesvdjSetMaxSweeps(
        gesvdj_params,
        max_sweeps);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cudaStat2 = cudaMalloc ((void**)&d_S2   , sizeof(dt)*econ_s2);  // 20 
    cudaStat3 = cudaMalloc ((void**)&d_U2   , sizeof(dt)*ldu2*econ_s2);  // 400 * 20  
    cudaStat4 = cudaMalloc ((void**)&d_V2   , sizeof(dt)*ldv2*econ_s2);  // 20*20
    //cudaStat4 = cudaMalloc ((void**)&d_VT2  , sizeof(dt)*ldv2*econ_s2);  // 20*20
    //cudaStat4 = cudaMalloc ((void**)&d_G3   , sizeof(dt)*ldv2*econ_s2);  // 20*20
    cudaStat5 = cudaMalloc ((void**)&d_info2, sizeof(int));
    // cudaStat1 = cudaHostAlloc((void**)&h_G2 , sizeof(dt)*ldu2*econ_s2, 0);
    // cudaStat1 = cudaHostAlloc((void**)&h_G3 , sizeof(dt)*ldv2*econ_s2, 0);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);
    assert(cudaSuccess == cudaStat5);
    //printf("step3 初始化 ------------------ \n");
    // cout<< " m2:" << m2 << " n2:" << n2  <<endl;
/* step 4: query workspace of SVD */
    status = cusolverDnSgesvdj_bufferSize(
        cusolverH,
        jobz, /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
              /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
        econ, /* econ = 1 for economy size */
        m2,    /* nubmer of rows of A, 0 <= m */
        n2,    /* number of columns of A, 0 <= n  */
        d_A2,  /* m-by-n */
        lda2,  /* leading dimension of A */
        d_S2,  /* min(m,n) */
              /* the singular values in descending order */
        d_U2,  /* m-by-m if econ = 0 */
              /* m-by-min(m,n) if econ = 1 */
        ldu2,  /* leading dimension of U, ldu >= max(1,m) */
        d_V2,  /* n-by-n if econ = 0  */
              /* n-by-min(m,n) if econ = 1  */
        ldv2,  /* leading dimension of V, ldv >= max(1,n) */
        &lwork2,
        gesvdj_params);

    assert(CUSOLVER_STATUS_SUCCESS == status);
    //printf("svd分配空间 ------------------ \n");
    // cout << "-------------------------\npart shree svd work space: " << sizeof(dt) * lwork2 << endl;
    cudaStat1 = cudaMalloc((void**)&d_work2 , sizeof(dt)*lwork2);
    assert(cudaSuccess == cudaStat1);
    //printf("svd 初始化d_works ------------------ \n");
    /* step 5: compute SVD */

    // GPUTimer timer3;
    // timer3.start();
    status = cusolverDnSgesvdj(
        cusolverH,
        jobz,  /* CUSOLVER_EIG_MODE_NOVECTOR: compute singular values only */
               /* CUSOLVER_EIG_MODE_VECTOR: compute singular value and singular vectors */
        econ,  /* econ = 1 for economy size */
        m2,     /* nubmer of rows of A, 0 <= m */
        n2,     /* number of columns of A, 0 <= n  */
        d_A2,   /* m-by-n */
        lda2,   /* leading dimension of A */
        d_S2,   /* min(m,n)  */
               /* the singular values in descending order */
        d_U2,   /* m-by-m if econ = 0 */
               /* m-by-min(m,n) if econ = 1 */
        ldu2,   /* leading dimension of U, ldu >= max(1,m) */
        d_V2,   /* n-by-n if econ = 0  */
               /* n-by-min(m,n) if econ = 1  */
        ldv2,   /* leading dimension of V, ldv >= max(1,n) */
        d_work2,
        lwork2,
        d_info2,
        gesvdj_params);
    cudaStat1 = cudaDeviceSynchronize();
    // time = timer3.seconds();
    // cout << "----------------------------------\npart three svd run time: " << time << endl;
    //printf("step svd计算 ------------------ \n");
    assert(CUSOLVER_STATUS_SUCCESS == status);
    assert(cudaSuccess == cudaStat1);


    // printMatrix_Device(ldu2,econ_s2,d_U2,ldu2,"U2");
    // printMatrix_Device(econ_s2,1,d_S2,econ_s2,"S2");
    // printMatrix_Device(ldv2,econ_s2,d_V2,ldv2,"V2");
    
    if (d_A2) cudaFree(d_A2);
    if ( 0 == info2 ){
        printf("gesvdj converges \n");
    }else if ( 0 > info2 ){
        printf("%d-th parameter is wrong \n", -info2);
        exit(1);
    }else{
        printf("WARNING: info = %d : gesvdj does not converge \n", info2 );
    }
    

    status = cusolverDnXgesvdjGetSweeps(
        cusolverH,
        gesvdj_params,
        &executed_sweeps);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    status = cusolverDnXgesvdjGetResidual(
        cusolverH,
        gesvdj_params,
        &residual);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    printf("residual |A - U*S*V**H|_F = %E \n", residual );
    printf("number of executed sweeps = %d \n", executed_sweeps );

      
//计算二范数。获得截断值
    // time = 200.0f;
    // GPUTimer timer2;
    // timer2.start();
    dt *d_Shardm2 = NULL;
    cudaStat1 = cudaMalloc((void**)&d_Shardm2, sizeof(dt)*econ_s2);
    assert(cudaStat1 == cudaSuccess);
    cublas_status = cublasSnrm2(handle, econ_s2,
                d_S2, 1, &norm_d_S);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaStat1 == cudaSuccess);
    truncated_Value = norm_d_S * ep * norm_d_S * ep;
///获得截断索引 truncat_number 也就是h_r_array
///TODO
    // printMatrix_Device(econ_s,1,d_S,econ_s,"d_S");
    // cublas_status = cublasSdot (handle, econ_s,
    //                        d_S, econ_s,
    //                        d_S, econ_s,
    //                        d_Sdot);
    // assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    // cudaStat1 = cudaDeviceSynchronize();
    // assert(cudaStat1 == cudaSuccess);
    // printMatrix_Device(econ_s,1,d_Sdot,econ_s,"d_Sdot");
    sumError = 0.0f;
    truncat_number_test = 0;
    hardm<<<blocks,threads>>>(d_S2,d_S2,d_Shardm2,econ_s2);
    // printMatrix_Device(econ_s2,1,d_S2,econ_s2,"S2");
    // printMatrix_Device(econ_s,1,d_Shardm,econ_s,"Shardm");
    cout << "truncated_Value2" << truncated_Value << endl;
    while(sumError < truncated_Value){
        cublas_status = cublasSasum(handle, truncat_number_test,
                            d_Shardm2, 1, &sumError);
        assert(cublas_status == CUBLAS_STATUS_SUCCESS);
        // cout << "sumError2" << sumError << endl;
        truncat_number_test++;
    }
    // time = timer2.seconds();
    // cout << "truncat_number_test2:" << econ_s2 - truncat_number_test + 3 <<endl;
    if(d_Shardm2) cudaFree(d_Shardm2);

     // cout <<"truncated :" << truncated  << " norm_d_S:" << norm_d_S << endl;
///获得截断索引 truncat_number 也就是h_r_array
///TODO
    h_r_array[2] = econ_s2 - truncat_number_test + 2;
    // h_r_array[2] = min(econ_s2 - truncat_number_test + 2, h_r_array[2]);
    truncat_number = h_r_array[2];
    cout << "truncat_number:" << truncat_number << endl;


    // float *d_Struncat2 = NULL;//, *d_Vtruncat2 = NULL;
    // cudaStat1 = cudaMalloc((void**)&d_Struncat2, sizeof(float)*truncat_number);
    // cudaStat2 = cudaMalloc((void**)&d_Vtruncat2, sizeof(float)*truncat_number*n2);
    cudaStat3 = cudaMalloc ((void**)&d_VT2  , sizeof(dt)*ldv2*econ_s2);  // 20*20
    cudaStat4 = cudaMalloc ((void**)&d_G3   , sizeof(dt)*ldv2*truncat_number);  // 20*20
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
    assert(cudaStat3 == cudaSuccess);
    assert(cudaStat4 == cudaSuccess);
    cudaStat1 = cudaHostAlloc((void**)&h_G2 , sizeof(dt)*ldu2*truncat_number, 0);
    cudaStat2 = cudaHostAlloc((void**)&h_G3 , sizeof(dt)*ldv2*truncat_number, 0);
    assert(cudaStat1 == cudaSuccess);
    assert(cudaStat2 == cudaSuccess);
///截断S2
    // cublas_status = cublasScopy(handle, truncat_number,
    //                         d_S2, 1,
    //                         d_Struncat2, 1);
    // assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    // cudaStat1 = cudaDeviceSynchronize();
    // assert(cudaStat1 == cudaSuccess);
    // if(d_S2) cudaFree(d_S2);
///截断V
    // printMatrix_Device(truncat_number, 1, d_Struncat2, truncat_number, "Struncat2");
    // printMatrix_Device(n, econ_s, d_V, n, "V");
    // cout << 
    // cublas_status = cublasScopy(handle, truncat_number*n2,
    //                         d_V2, 1,
    //                         d_Vtruncat2, 1);
    // assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    // cudaStat1 = cudaDeviceSynchronize();
    // assert(cudaStat1 == cudaSuccess);
    // if(d_V2) cudaFree(d_V2);
    // printMatrix_Device(n2, truncat_number, d_Vtruncat2, n2, "Vtruncat2");

    // GPUTimer timer4;
    // timer4.start();
    //获得最后的h_G3 先对V进行转置
    cublas_status = cublasSgeam(handle,
                                CUBLAS_OP_T, CUBLAS_OP_N, 
                                econ_s2, ldv2,
                                &alpha,
                                d_V2, ldv2,
                                &beta,
                                d_V2, ldv2,
                                d_VT2, econ_s2);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);

    cublas_status = cublasSdgmm(handle,
                                CUBLAS_SIDE_LEFT,
                                truncat_number, ldv2,
                                d_VT2, econ_s2,
                                d_S2, 1,
                                d_G3, truncat_number);
    assert(cublas_status == CUBLAS_STATUS_SUCCESS);
    cudaStat1 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);
    // time = timer4.seconds();
    // cout << "----------------------------------\npart four G3 run time: " << time << endl;

    cudaStat1 = cudaMemcpyAsync(&info2, d_info2, sizeof(int), cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpyAsync(h_G2, d_U2, sizeof(dt)*ldu2*truncat_number, cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpyAsync(h_G3, d_G3, sizeof(dt)*ldv2*truncat_number, cudaMemcpyDeviceToHost);
    cudaStat4 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);
    comp_rat_number += (ldu2*truncat_number + ldv2*truncat_number);
    float comp_ratio =  h_n_array[0]*h_n_array[1]*h_n_array[2]*1.0 / comp_rat_number*1.0;
    cout << "comp_ratio: " << comp_ratio << endl;
    
    int calAllNumber = h_n_array[0]*h_n_array[1]*h_n_array[2];
    dim3 block3n((calAllNumber+1024-1)/1024,1,1); 
    float *d_coreG2G3 = NULL, *d_coreG1G2G3;
    cudaMalloc((void**)&d_coreG2G3, sizeof(float)*ldu2*ldv2);
    cudaMalloc((void**)&d_coreG1G2G3, sizeof(float)*calAllNumber);

    // const float alpha = 1.0;
    // const float beta = 0.0;
    cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N,
                           ldu2, ldv2, truncat_number,
                           &alpha,
                           d_U2, ldu2,
                           d_G3, truncat_number,
                           &beta,
                           d_coreG2G3, ldu2);
    cublasSgemm(handle,CUBLAS_OP_N, CUBLAS_OP_N,
                           h_n_array[0],  h_n_array[1]* h_n_array[2], h_r_array[1],
                           &alpha,
                           d_U,  h_n_array[0],
                           d_coreG2G3,  h_r_array[1],
                           &beta,
                           d_coreG1G2G3, h_n_array[0]);
    cudaDeviceSynchronize();
    sub<<<block3n,threads>>>(d_A,d_coreG1G2G3,h_n_array[0],h_n_array[1],h_n_array[2]);
    cudaDeviceSynchronize();

    float re = 0.0, before = 0.0;
    cublasSnrm2(handle,calAllNumber,d_coreG1G2G3,1,&re);
    cublasSnrm2(handle,calAllNumber,d_A,1,&before);
    cudaDeviceSynchronize();

    cout << "re: "<< re << " before: " << before <<endl;
    cout<<"error rate: "<< re/before<<endl;

    if(d_A) cudaFree(d_A);
    if(d_G3) cudaFree(d_G3);
    if(d_U) cudaFree(d_U);
    if(d_coreG2G3) cudaFree(d_coreG2G3);
    if(d_coreG1G2G3) cudaFree(d_coreG1G2G3);

    if (d_S2) cudaFree(d_S2);
    if (d_U2) cudaFree(d_U2);
    if (d_V2) cudaFree(d_V2);
    if (d_VT2) cudaFree(d_VT2);
    if (d_info2) cudaFree(d_info2);
    if (d_work2 ) cudaFree(d_work2);

    if (h_G1) cudaFreeHost(h_G1);
    if (h_G2) cudaFreeHost(h_G2);
    if (h_G3) cudaFreeHost(h_G3);

    if (cusolverH) cusolverDnDestroy(cusolverH);
    if (stream      ) cudaStreamDestroy(stream);
    if (gesvdj_params) cusolverDnDestroyGesvdjInfo(gesvdj_params);
    // time = timer0.seconds();
    // cout << "----------------------------------\nall run time: " << time << endl;

}