/*************************************************************************
	> File Name: func.h
	> Author: honghao
	> Mail: honghao@shu.edu.cn 
	> Created Time: 2020年04月2日 星期四 14时25分17秒
 ************************************************************************/
#include "head.h"

__global__  void floattohalf(dt *AA,half *BB,long m){
    long i = blockIdx.x*blockDim.x+threadIdx.x;
    const long temp = blockDim.x*gridDim.x;
    if(i<m){
        BB[i] = __float2half(AA[i]);
        i+=temp;
    }
    __syncthreads();
}

__global__ void matrixInvertColumn(dt *d_A, dt *d_Ainv, int m, int n){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<m*n){
        int row = i%m;
        int col = i/m;
        d_Ainv[row+(n-col-1)*m]=d_A[i];
        // d_Ainv[i] = d_A[];
    }
    __syncthreads();
}

void f2h(dt *A,half *B,long num){
    dim3 threads(1024,1,1);
    dim3 blocks((num+1024-1)/1024,1,1);   
    floattohalf<<<blocks,threads>>>(A,B,num);
}

void genTRTensor(dt *T,long a,long b,long c,long r){
    dt *AA,*BB,*CC;    
    cudaHostAlloc((void**)&AA,sizeof(dt)*a*r,0);
    cudaHostAlloc((void**)&BB,sizeof(dt)*b*r,0);
    cudaHostAlloc((void**)&CC,sizeof(dt)*c*r,0);
    for(long i = 0;i<a*r;i++){
        AA[i]=rand()*0.1/(RAND_MAX*0.1);
    }
    for(long i = 0;i<b*r;i++){
        BB[i]=rand()*0.1/(RAND_MAX*0.1);
    }
    for(long i = 0;i<c*r;i++){
        CC[i]=rand()*0.1/(RAND_MAX*0.1);
    }
    dt *d_T,*d_CC,*d_BB,*d_AA;
    cudaMalloc((void**)&d_AA,sizeof(dt)*a*r);
    cudaMalloc((void**)&d_BB,sizeof(dt)*b*r);
    cudaMalloc((void**)&d_CC,sizeof(dt)*c*r);
    cudaMalloc((void**)&d_T,sizeof(dt)*a*b*c);
    cudaMemcpyAsync(d_BB,BB,sizeof(dt)*b*r,cudaMemcpyHostToDevice,0);
    cudaMemcpyAsync(d_CC,CC,sizeof(dt)*c*r,cudaMemcpyHostToDevice,0);
    cudaMemcpyAsync(d_AA,AA,sizeof(dt)*a*r,cudaMemcpyHostToDevice,0);
    dt *d_CKRB;
    cudaMalloc((void**)&d_CKRB,sizeof(dt)*c*r*b);
    cudaDeviceSynchronize();

    // printMatrix_Device(a,r,d_AA,a,"AA");
    // printMatrix_Device(b,r,d_BB,b,"BB");
    // printMatrix_Device(c,r,d_CC,c,"CC");

    //X1=A*(CkrB)'  a*r  r*(bc)
    dt alpha = 1.0;
    dt beta = 0.0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemmStridedBatched(handle,CUBLAS_OP_N,CUBLAS_OP_T,
    		b,c,1,
    		&alpha,
    		d_BB,b,b,
    		d_CC,c,c,
    		&beta,d_CKRB,
    		b,b*c,r);
    // printMatrix_Device(b,c*r,d_CKRB,b,"CkrB");

    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,b*c,r,&alpha,d_AA,a,d_CKRB,b*c,&beta,d_T,a);
    // printMatrix_Device(a,b*c,d_T,a,"T");

    cudaMemcpyAsync(T,d_T,sizeof(dt)*a*b*c,cudaMemcpyDeviceToHost,0);
    cudaDeviceSynchronize();

    cudaFree(d_AA);
    cudaFree(d_BB);
    cudaFree(d_CC);
    cudaFree(d_CKRB);
    cudaFree(d_T);
    cudaFreeHost(AA);
    cudaFreeHost(BB);
    cudaFreeHost(CC);
    cublasDestroy(handle);
}
__global__ void warmup()
{
    int sum=0;
    for(int i = 0; i < 1000; i++) {
        sum+=i;
    }
}   

void warmupcu(){
    warmup<<<1,1>>>();
}

void printMatrix(int m, int n, const dt*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            float Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        }
    }
    cout << " ------------------------------------" << endl;
}

void printMatrix_Device(int m, int n, const dt*d_A, int lda, const char* name)
{
    dt *h_A = NULL;
    cudaHostAlloc((void**)&h_A , sizeof(dt)*m*n, 1);
    cudaMemcpy(h_A, d_A, sizeof(dt)*m*n, cudaMemcpyDeviceToHost);
    printMatrix(m, n, h_A, lda, name);
    if(h_A) cudaFreeHost(h_A);
}

void printTensor(const dt *d_des,long m,long n,long l,const char* name){
    cout << name << endl;
    dt *des = new dt[m*n*l]();
    cudaMemcpy(des,d_des,sizeof(dt)*m*n*l,cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int k = 0;k<l;k++){
        for(int i = 0;i<n;i++){
            for(int j = 0;j<m;j++){
                printf("%s(%d,%d,%d)=%f\n",name,j+1,i+1,k+1,des[k*m*n+i*m+j]);
                // cout<<des[k*m*n+i*m+j]<<" ";
            }
            cout<<endl;
        }
        // cout<<"~~~~~~~~~~~~~~~~"<<endl;
    }
 
    delete[] des;des=nullptr;
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
        T2[k*col*m+row*k+tube] = T1[tube*m*n+col*m+row];
    }
    __syncthreads();
}

__global__ void tensorToMode231(dt *T1,dt *T2,int m,int n,int k){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<m*n*k){
        int tube = i/(m*n);
        int row = (i-tube*(m*n))%m;
        int col = (i-tube*(m*n))/m;
        T2[row*k*n+tube*n+col] = T1[tube*m*n+col*m+row]; //T1[i]
    }
    __syncthreads();
}

__global__ void elemin(dt *A,dt *B, long n){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<n){
		B[i] = A[i] - B[i];	
	}
    __syncthreads();
}

/*
 * Creates the "inverted" sigma matrix starting from the vector of singular values
 *
 */
__global__ void invert_sigma(float * d_S, float * d_Sinv, int n) {
    
    float myeps = 0.001;
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    //# Soft-Thresholding
    
    if (i < n) {
        //### TODO must be done outside
        //### Fill the line with zeros
        for (int j = 0; j < n; j++) {
            d_Sinv[i*n + j] = 0;
        }
        
        if (d_S[i] > d_S[0]*myeps) {
            d_Sinv[i*n + i] = 1/d_S[i];
        } else {
            d_Sinv[i*n + i] = 0;
        }
    }
    __syncthreads();
}

/**
 *
 * d_X : the matrix whose pseudoinverse must be computed
 * n : the number of rows of the matrix
 * p : the number of columns of the matrix
 * d_Xpinv : the pseudoinverse of d_X
 */
int pinv(cublasHandle_t cb_handle, cusolverDnHandle_t cs_handle, float * d_X, int n, int p, float * d_Xpinv, cublasGemmAlgo_t algo) {
    if (n < p) {
        cout << "n must be greater or equal than p; aborting." << endl;
        return -1;
    }
    
    float alpha = 1.0f;
    float beta = 0.0f;
    int np = n*p;
    cudaError_t cudaStat1_pinv = cudaSuccess;
    cudaError_t cudaStat2_pinv = cudaSuccess;
    cudaError_t cudaStat3_pinv = cudaSuccess;
    cudaError_t cudaStat4_pinv = cudaSuccess;
    cudaError_t cudaStat5_pinv = cudaSuccess;
    cublasStatus_t cublas_status_pinv = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status_pinv= CUSOLVER_STATUS_SUCCESS;
    
    dim3 threadsPerBlock(512,1,1);
    dim3 numBlocks((p+512-1)/512,1,1); //for invert_sigma
    
    //### used to control the level of debug output
    int h_lwork = 0, *devInfo = 0;
    float *d_S = 0, *d_U = 0, *d_VH = 0, *d_Sinv = 0, * d_work = 0;
    cudaStat1_pinv = cudaMalloc ((void**)&d_S  , sizeof(float)*p);
    cudaStat2_pinv = cudaMalloc ((void**)&d_U  , sizeof(float)*np);
    cudaStat3_pinv = cudaMalloc ((void**)&d_VH , sizeof(float)*p*p);
    cudaStat4_pinv = cudaMalloc ((void**)&d_Sinv , sizeof(float)*p*p);
    cudaStat5_pinv = cudaMalloc ((void**)&devInfo, sizeof(int));
    assert(cudaSuccess == cudaStat1_pinv);
    assert(cudaSuccess == cudaStat2_pinv);
    assert(cudaSuccess == cudaStat3_pinv);
    assert(cudaSuccess == cudaStat4_pinv);
    assert(cudaSuccess == cudaStat5_pinv);
    
    cusolver_status_pinv = cusolverDnSgesvd_bufferSize(cs_handle, n, p, &h_lwork);
    assert(cusolver_status_pinv == CUSOLVER_STATUS_SUCCESS);

    cudaStat1_pinv = cudaMalloc((void**)&d_work , sizeof(float)*h_lwork);
    assert(cudaSuccess == cudaStat1_pinv);

    //compute the SVD
    cusolver_status_pinv = cusolverDnSgesvd(cs_handle, 'S', 'S', n, p, d_X, n, d_S, d_U, n, d_VH, p, d_work, h_lwork, NULL, devInfo);
    cudaStat1_pinv = cudaDeviceSynchronize();
    assert(cusolver_status_pinv == CUBLAS_STATUS_SUCCESS);
    assert(cudaSuccess == cudaStat1_pinv);
    // printMatrix_Device(n, p, d_U, n, "U");
    // printMatrix_Device(p, 1, d_S, p, "S");
    // printMatrix_Device(p, p, d_VH, p, "VH");

    invert_sigma<<<numBlocks, threadsPerBlock>>>(d_S, d_Sinv, p);
    // printMatrix_Device(p, p, d_Sinv, p, "Sinv");
    //CUBLAS_OP_C 共轭转置操作
    // cublas_status_pinv = cublasSgemm(cb_handle, CUBLAS_OP_T, CUBLAS_OP_T, p, p, p, &alpha, d_VH, p, d_Sinv, p, &beta, d_Sinv, p);
    cublas_status_pinv = cublasGemmEx(cb_handle, CUBLAS_OP_T, CUBLAS_OP_T, p, p, p,
                            &alpha, d_VH, CUDA_R_32F, p, d_Sinv, CUDA_R_32F, p,
                            &beta, d_Sinv, CUDA_R_32F, p,
                            CUDA_R_32F, algo);
    assert(cublas_status_pinv == CUBLAS_STATUS_SUCCESS);
    // printMatrix_Device(p, p, d_Sinv, p, "Sinv");
    // printMatrix_Device(n, p, d_U, n, "U2");
    //d_Xpinv = d_Sinv x d_U^H
    // cublas_status_pinv = cublasSgemm(cb_handle, CUBLAS_OP_N, CUBLAS_OP_T, p, p, n, &alpha, d_Sinv, p, d_U, n, &beta, d_Xpinv, p);
    cublas_status_pinv = cublasGemmEx(cb_handle, CUBLAS_OP_N, CUBLAS_OP_T, p, n, p,
                            &alpha, d_Sinv, CUDA_R_32F, p, d_U, CUDA_R_32F, n,
                            &beta, d_Xpinv, CUDA_R_32F, p,
                            CUDA_R_32F, algo);
    assert(cublas_status_pinv == CUBLAS_STATUS_SUCCESS);
    // printMatrix_Device(p, n, d_Xpinv, p, "Xpinv");
    // cudaStat1_pinv = cudaDeviceSynchronize();
    // assert(cudaSuccess == cudaStat1_pinv);

    cudaFree(d_work);
    cudaFree(devInfo);
    
    cudaFree(d_S);
    cudaFree(d_U);
    cudaFree(d_VH);
    cudaFree(d_Sinv);
    return 0;
}