#include "head.h"
void per_lmr(dt *d_L,int a,int b,int c,int d,int e,int f)
{	
	cudaDataType_t typeA = CUDA_R_32F;
    cudaDataType_t typeC = CUDA_R_32F;
    cudaDataType_t typeCompute = CUDA_R_32F;

    std::vector<int> modeA{'a','b','c','d','e','f'};
    std::vector<int> modeC{'b','c','f','a','d','e'};

    int nmodeA = modeA.size();
    int nmodeC = modeC.size();
    std::unordered_map<int, int64_t> extent;

    extent['a'] = a;
    extent['b'] = b;
    extent['c'] = c;
    extent['d'] = d;
    extent['e'] = e;
    extent['f'] = f;

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
float lanczos(dt *d_psiGround,dt *d_L,dt *d_M,dt *d_R,int maxit,int krydim,int size_psi,int *sizeL,int *sizeA,int *sizeR,cublasHandle_t handle,cusolverDnHandle_t cusolverH,cutensorHandle_t tensor_handle)
{	
	dt *d_A;
	dt *A = new dt[krydim*krydim];//为了方便后面的元素赋值
	dt temp;
	dt *p[krydim+1]; //存储d_psi 中的每一列（krydim + 1）
	for(int k=0;k<krydim+1;k++)
	{
		cudaMalloc((void**)&p[k],sizeof(dt)*size_psi);
	}
	dt *d_ig_temp;
	dt *d_W; //用于存储A的特征向量
	dt *d_psi_temp; //用于与A 的特征向量相乘
	dt *p_temp = new dt[size_psi];
	dt w=0.0;
	cudaMalloc((void**)&d_A,sizeof(dt)*krydim*krydim);
	cudaMalloc((void**)&d_W,sizeof(dt)*krydim);
	cudaMalloc((void**)&d_psi_temp,sizeof(dt)*size_psi*krydim);
	cudaMalloc((void**)&d_ig_temp,sizeof(dt)*size_psi);

	// 循环 maxit 次
	dim3 threads(1024,1,1);
	dim3 blockR((sizeR[0]*sizeR[1]*sizeR[2]+1024-1)/1024,1,1);
	dim3 blockL((sizeL[0]*sizeL[1]*sizeL[2]+1024-1)/1024,1,1);
	dim3 blockM((2*2*4*4+1024-1)/1024,1,1);
	dt sh=0.0;

    float alpha = 1.0;
    float beta = 0.0;
    //float beta2 = -1.0;

    for(int j=0;j<1;j++)
    {
		sh=norm(d_psiGround,sh,size_psi,handle);
		if(sh < 1e-8)
		{
			sh = 1e-8;
		}	
		sh=1/sh;
		division(d_psiGround,sh,d_psiGround,size_psi,1,1,handle);
	  	cudaDeviceSynchronize();

	  	dt *d_L_mode3,*d_M_mode3;
		cudaMalloc((void**)&d_L_mode3,sizeof(dt)*sizeL[0]*sizeL[1]*sizeL[2]);
		cudaMalloc((void**)&d_M_mode3,sizeof(dt)*2*2*4*4);

		dt *d_L_mode3T,*d_M_mode3T;
		cudaMalloc((void**)&d_L_mode3T,sizeof(dt)*sizeL[0]*sizeL[1]*sizeL[2]);
		cudaMalloc((void**)&d_M_mode3T,sizeof(dt)*2*2*4*4);



		tensorToMode3<<<blockL,threads>>>(d_L,d_L_mode3,sizeL[0],sizeL[1],sizeL[2]);
		cudaDeviceSynchronize();
		transpose<<<blockL,threads>>>(d_L_mode3,d_L_mode3T,sizeL[2],sizeL[0]*sizeL[1]);
		cudaDeviceSynchronize();

		tensorToMode3<<<blockM,threads>>>(d_M,d_M_mode3,2*2,4,4);
		cudaDeviceSynchronize();
		transpose<<<blockM,threads>>>(d_M_mode3,d_M_mode3T,4,2*2*4);
		cudaDeviceSynchronize();

		dt *d_R_mode3;
		cudaMalloc((void**)&d_R_mode3,sizeof(dt)*sizeR[0]*sizeR[1]*sizeR[2]);
		dt *d_R_mode3T;
		cudaMalloc((void**)&d_R_mode3T,sizeof(dt)*sizeR[0]*sizeR[1]*sizeR[2]);
		tensorToMode3<<<blockR,threads>>>(d_R,d_R_mode3,sizeR[0],sizeR[1],sizeR[2]);
		cudaDeviceSynchronize();
		transpose<<<blockR,threads>>>(d_R_mode3,d_R_mode3T,sizeR[2],sizeR[0]*sizeR[1]);
		cudaDeviceSynchronize();

	  	p[0] = d_psiGround;
	  	 	 //p[ip]=(p[ip-1],L,M,R)
	  		/************************
	  		* ncon(p[ip],L,M,R) 
	  		* 
	  		* d_psiGround * (L*M*R)
	  		************************/
	  		// L * M -> [1,-1,-2,3,4]
	  	dt *d_LM;
	  	cudaMalloc((void**)&d_LM,sizeof(dt)*sizeL[0]*sizeL[1]*2*2*4);	  	
	  	cublasSgemm(handle,
	  		        CUBLAS_OP_N,CUBLAS_OP_T,
	  		        sizeL[0]*sizeL[1],2*2*4,sizeL[2],
	  		        &alpha,d_L,sizeL[0]*sizeL[1],d_M,2*2*4,
	  		        &beta,d_LM,sizeL[0]*sizeL[1]
	  		        );
  
		cudaDeviceSynchronize();  	

	  	//LM * R -> [1,-1,-2,3,5,-3]
	  	dt *d_LMR;
	  	cudaMalloc((void**)&d_LMR,sizeof(dt)*sizeL[0]*sizeL[1]*2*2*sizeR[0]*sizeR[1]);	  	
	  	cublasSgemm(handle,
	  		        CUBLAS_OP_N,CUBLAS_OP_T,
	  		        sizeL[0]*sizeL[1]*2*2,sizeR[0]*sizeR[1],sizeR[2],
	  		        &alpha,d_LM,sizeL[0]*sizeL[1]*2*2,d_R,sizeR[0]*sizeR[1],
	  		        &beta,d_LMR,sizeL[0]*sizeL[1]*2*2
	  		        );
       
	  	// LMR *psiGround -> [-1,-2,-3]

	  	//[-1,1,-2,3,-3,5] -> [-1,-2,-3,1,3,5]
	  	per_lmr(d_LMR,sizeL[0],sizeL[1],2,2,sizeR[0],sizeR[1]);
	 	  
	  	for(int ip =1;ip<krydim+1;ip++)
	  	{	  			  		
	  		// LMR(permute之后) * p[ip]	  			  		  		
	  		cublasSgemm(handle,
	  		            CUBLAS_OP_N,CUBLAS_OP_N,
	  		            sizeL[1]*2*sizeR[1],1,sizeL[0]*2*sizeR[0],
	  		            &alpha,d_LMR,sizeL[1]*2*sizeR[1],p[ip-1],sizeL[0]*2*sizeR[0],
	  		            &beta,p[ip],sizeL[1]*2*sizeR[1]
	 		            );	 	
	 		            
	  		cudaDeviceSynchronize();
	  			  		
	  		for(int ig=0;ig<ip;ig++)
	  		{
	  			temp = cuda_dot(p[ip],p[ig],size_psi,temp,handle);
	  			//cout<<ip<<"zhong jian bianliang :"<<temp<<endl;  
	  			A[(ip-1)*4+ig]=temp;
	  			A[(ig*4+(ip-1))]=temp;
	  		}
	  		for(int ig=0;ig<ip;ig++)
	  		{
	  			
	  			//p[ip] = p[ip] - cuda_dot(ig,ip)*ig
	  			temp = cuda_dot(p[ig],p[ip],size_psi,temp,handle);

	  			dt beta2 = -temp;

	  			cublasSgeam(handle,CUBLAS_OP_N,CUBLAS_OP_N,
	  			            size_psi,1,
	  			            &alpha,p[ip],size_psi,
	  			            &beta2,p[ig],size_psi,p[ip],size_psi
	  			            );

	  			cudaDeviceSynchronize();
	  			sh = norm(p[ip],sh,size_psi,handle);
	  			//cout<<"ip:"<<ip<<", ig:"<<ig<<":p[ip] zheng jiao :"<<sh<<endl;

	  			if(sh < 1e-8)
	  			{
	  				sh = 1e-8;
	  			}	  			
	  			sh = 1/sh;
	  			division(p[ip],sh,p[ip],size_psi,1,1,handle);		  				  		  			
	  		}	
	  	}
	  	cudaFree(d_LM);
	  	cudaFree(d_LMR);
	  	cudaFree(d_R_mode3);
	  	cudaFree(d_M_mode3);
	  	cudaFree(d_R_mode3T);
	  	cudaFree(d_M_mode3T); 	
	  	cudaFree(d_L_mode3);
		cudaFree(d_L_mode3T);

	  	cudaMemcpy(d_A,A,sizeof(dt)*krydim*krydim,cudaMemcpyHostToDevice);
/*
	  	cout<<"sheng cheng de matrix A:"<<endl;
	  	for(int i=0;i<krydim*krydim;i++)
	  	{
	  		cout<<A[i]<<" _ ";
	  	}
	  	cout<<endl;
*/
	  	
	  	// 对 A 进行eig分解
	  	eig(d_A,d_W,krydim);

	  	cublasScopy(handle,size_psi,p[0],1,d_psi_temp,1);
	  	cublasScopy(handle,size_psi,p[1],1,d_psi_temp+size_psi,1);
	  	cublasScopy(handle,size_psi,p[2],1,d_psi_temp+2*size_psi,1);
	  	cublasScopy(handle,size_psi,p[3],1,d_psi_temp+3*size_psi,1);
	  	cudaDeviceSynchronize();

	  	cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
	  	            size_psi,1,krydim,
	  	            &alpha,d_psi_temp,size_psi,d_A,krydim,
	  	            &beta,d_psiGround,size_psi
	  	            );
	  	cudaDeviceSynchronize();
	  }

	sh=norm(d_psiGround,sh,size_psi,handle);
	if(sh < 1e-8)
	{
		sh = 1e-8;
	}	
	division(d_psiGround,sh,d_psiGround,size_psi,1,1,handle);
	cudaDeviceSynchronize();  

	
	cudaMemcpy(&w,d_W,sizeof(dt)*1,cudaMemcpyDeviceToHost);
	//cout<<" w is :"<<w<<endl;

/*	
	cout<<"w eig_values is :"<<endl;
	for(int i=0;i<krydim;i++)
	{
		cout<<w[i]<<"_";
	}
	cout<<endl<<"~~~~~~~~~~~~~~~~"<<endl;
*/		
	cudaFree(d_A);
	cudaFree(d_W);
	cudaFree(d_psi_temp);
	cudaFree(p[0]);cudaFree(p[1]);cudaFree(p[2]);cudaFree(p[3]);cudaFree(p[4]);

	return w;

}