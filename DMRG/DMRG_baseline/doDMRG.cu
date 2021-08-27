#include "head.h"
void doDMRG_MPO(dt *M,dt **d_A,dt *ML,dt *MR,int (*sizeA)[3],int numsweep,int dispon,bool updateon,int maxit,int krydim,int hh)
{   
	int Nsite = hh;

	cusolverDnHandle_t cusolverH = NULL;
	cusolverDnCreate(&cusolverH); 

	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSetMathMode(handle,CUBLAS_TENSOR_OP_MATH);

	cutensorHandle_t tensor_handle;
    cutensorInit(&tensor_handle); 

	dt alpha = 1.0;
	dt beta =0.0;
	dt norm_temp;
	// L R分别有 Nsite个
	dt *d_L[Nsite],*d_R[Nsite];
	int sizeL[Nsite][3];
	int sizeR[Nsite][3];
	sizeL[0][0] = 1;
	sizeL[0][1] = 1;
	sizeL[0][2] = 4;

	sizeR[Nsite-1][0] = 1;
	sizeR[Nsite-1][1] = 1;
	sizeR[Nsite-1][2] = 4;

	cudaMalloc((void**)&d_L[0],sizeof(dt)*4);
	cudaMalloc((void**)&d_R[Nsite-1],sizeof(dt)*4);
	cudaMemcpy(d_L[0],ML,sizeof(dt)*4,cudaMemcpyHostToDevice);
	cudaMemcpy(d_R[Nsite-1],MR,sizeof(dt)*4,cudaMemcpyHostToDevice);

	dt *d_M;//2*2*4*4
	
	cudaMalloc((void**)&d_M,sizeof(dt)*64);
	cudaMemcpy(d_M,M,sizeof(dt)*64,cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

 clock_t t1,t2;
	double times=0.0;
t1=clock();	

	dt *d_sWeight[Nsite]; //size = sizeA[0][1] * sizeA[0][1]
	dt *d_QRweight[Nsite+1];
	//QR 分解进行正则化，其中，Q会直接放进A里
	for(int i=0;i<Nsite-1;i++)
	{		
		cudaMalloc((void**)&d_sWeight[i],sizeof(dt)*sizeA[i][2]*sizeA[i][2]);
		QR(d_A[i],d_sWeight[i],sizeA[i][0]*sizeA[i][1],sizeA[i][2],cusolverH);

		matmul(d_sWeight[i],d_A[i+1],sizeA[i+1][0],sizeA[i+1][1],sizeA[i+1][2],handle);
		//正则化  A/nomr(A)
		norm_temp = norm(d_A[i+1],norm_temp,sizeA[i+1][0]*sizeA[i+1][1]*sizeA[i+1][2],handle);
		norm_temp = 1/norm_temp;
		division(d_A[i+1],norm_temp,d_A[i+1],sizeA[i+1][0],sizeA[i+1][1],sizeA[i+1][2],handle);

	}
	//边缘处理 
	cudaMalloc((void**)&d_sWeight[Nsite-1],sizeof(dt)*sizeA[Nsite-1][2]*sizeA[Nsite-1][2]);
	QR(d_A[Nsite-1],d_sWeight[Nsite-1],sizeA[Nsite-1][0]*sizeA[Nsite-1][1],sizeA[Nsite-1][2],cusolverH);
	//正则化
	norm_temp = norm(d_sWeight[Nsite-1],norm_temp,sizeA[Nsite-1][2]*sizeA[Nsite-1][2],handle);
	norm_temp = 1/norm_temp;
	division(d_sWeight[Nsite-1],norm_temp,d_sWeight[Nsite-1],sizeA[Nsite-1][2],sizeA[Nsite-1][2],1,handle);
	d_QRweight[Nsite]=d_sWeight[Nsite-1];


	for(int i=0;i<Nsite-1;i++)
	{
		cudaMalloc((void**)&d_L[i+1],sizeof(dt)*sizeA[i][2]*sizeA[i][2]*4);
		//ncon(d_L[0],d_M,d_A[0],sizeA[0],sizeL[0],d_L[1],handle,tensor_handle);
		ncon(d_L[i],d_M,d_A[i],sizeA[i],sizeL[i],d_L[i+1],handle,tensor_handle);
		sizeL[i+1][0]=sizeA[i][2];
		sizeL[i+1][1]=sizeA[i][2];
		sizeL[i+1][2]=4;
	}	
	
	
  //从右到左 扫描执行  利用RQ分解
  // A4开始
  // A4 与 sWeight5 收缩为psiGround，用此来计算基态
  dim3 threads(1024,1,1);
 

	dt w=0.0;
	dt w2=100;
  dt *d_psiGround,*d_psiGround_mode1;
  dt *d_psiGround2,*d_psiGround2_mode1;
  int size_psi;
  for(int i = 0;i<numsweep;i++)
  {
  	// 从右到左
  	for(int p=Nsite-1;p>-1;p--)
  	{
  		 dim3 block0((sizeA[p][0]*sizeA[p][1]*sizeA[p][2]+1024-1)/1024,1,1);
  		// A(Nsite) 与 s(Nsite)收缩为 d_psiGround  	
  		cudaMalloc((void**)&d_psiGround,sizeof(dt)*sizeA[p][0]*sizeA[p][1]*sizeA[p][2]);//此变量为向量
  		cudaMalloc((void**)&d_psiGround_mode1,sizeof(dt)*sizeA[p][0]*sizeA[p][1]*sizeA[p][2]);
  		tensorToMode1<<<block0,threads>>>(d_psiGround,d_psiGround_mode1,sizeA[p][0],sizeA[p][1],sizeA[p][2]);

  		cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
  		            sizeA[p][0]*sizeA[p][1],sizeA[p][2],sizeA[p][2],
  		            &alpha,d_A[p],sizeA[p][0]*sizeA[p][1],d_QRweight[p+1],sizeA[p][2],
  		            &beta,d_psiGround,sizeA[p][0]*sizeA[p][1]
  		            );
  		cudaDeviceSynchronize();
  		
  		size_psi =sizeA[p][0]*sizeA[p][1]*sizeA[p][2];
 		
  		w=lanczos(d_psiGround,d_L[p],d_M,d_R[p],maxit,krydim,size_psi,sizeL[p],sizeA[p],sizeR[p],handle,cusolverH,tensor_handle);
  		cudaMalloc((void**)&d_QRweight[p],sizeof(dt)*sizeA[p][0]*sizeA[p][0]);
  		// 对更新后的 d_psiGround 做RQ分解
  		RQ(d_psiGround,d_A[p],d_QRweight[p],sizeA[p][0],sizeA[p][1],sizeA[p][2],handle,cusolverH);

  		norm_temp = norm(d_QRweight[p],norm_temp,sizeA[p][0]*sizeA[p][0],handle);
  		norm_temp = 1/norm_temp;
		division(d_QRweight[p],norm_temp,d_QRweight[p],sizeA[p][0],sizeA[p][0],1,handle);

  		if(p != 0){  			
  			sizeR[p-1][0]=sizeA[p][0]; sizeR[p-1][1]=sizeA[p][0]; sizeR[p-1][2]=4;
  			cudaMalloc((void**)&d_R[p-1],sizeof(dt)*sizeR[p-1][0]*sizeR[p-1][1]*sizeR[p-1][2]);
  			//更新R[p-1]
  			ncon2(d_R[p],d_M,d_A[p],sizeA[p],sizeR[p],d_R[p-1],handle,tensor_handle);
			}
  	}

  	// 从左到右
  	for(int p=0;p<Nsite;p++)
  	{	

  		dim3 block0((sizeA[p][0]*sizeA[p][1]*sizeA[p][2]+1024-1)/1024,1,1);
  		// A 与 d_QRweight 收缩
  		cudaMalloc((void**)&d_psiGround2,sizeof(dt)*sizeA[p][0]*sizeA[p][1]*sizeA[p][2]);
  		cudaMalloc((void**)&d_psiGround2_mode1,sizeof(dt)*sizeA[p][0]*sizeA[p][1]*sizeA[p][2]);
  		tensorToMode1<<<block0,threads>>>(d_psiGround2,d_psiGround2_mode1,sizeA[p][0],sizeA[p][1],sizeA[p][2]);


  		cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
  		            sizeA[p][0],sizeA[p][1]*sizeA[p][2],sizeA[p][0],
  		            &alpha,d_QRweight[p],sizeA[p][0],d_A[p],sizeA[p][0],
  		            &beta,d_psiGround2,sizeA[p][0]
  		            );  		
  		size_psi =sizeA[p][0]*sizeA[p][1]*sizeA[p][2];
	
		w=lanczos(d_psiGround2,d_L[p],d_M,d_R[p],maxit,krydim,size_psi,sizeL[p],sizeA[p],sizeR[p],handle,cusolverH,tensor_handle);

  		QR(d_psiGround2,d_QRweight[p+1],sizeA[p][0]*sizeA[p][1],sizeA[p][2],cusolverH);

  		cublasScopy(handle,sizeA[p][0]*sizeA[p][1]*sizeA[p][2],d_psiGround2,1,d_A[p],1);

  		norm_temp = norm(d_QRweight[p+1],norm_temp,sizeA[p][2]*sizeA[p][2],handle);
  		norm_temp = 1/norm_temp;
		division(d_QRweight[p+1],norm_temp,d_QRweight[p+1],sizeA[p][2],sizeA[p][2],1,handle);
		
  		if(p != Nsite-1)
  		{
  			ncon(d_L[p],d_M,d_A[p],sizeA[p],sizeL[p],d_L[p+1],handle,tensor_handle);
  		}
  	}

  	
  	if(abs(w2-w) < 1e-5)
  	{
  		cout<<"shou lian la :"<<i<<endl;
  		cout<<"w is :"<<w<<" ,w2 is:"<<w2<<endl;
  		break;
  	}

  	w2 = w;
  }

  t2=clock();
  times = (double)(t2-t1)/CLOCKS_PER_SEC;
  cout<<"cost time :"<<times<<"s"<<endl;

  /* 
	cudaFreeHost(L0);
	cudaFreeHost(R3);
	cudaFree(d_A1);
	cudaFree(d_A2);
	cudaFree(d_A3);
	cudaFree(d_A4);
	cudaFree(d_U1);
	cudaFree(d_S1);
	cudaFree(d_VH1);
	cublasDestroy(handle);

	cout<<"finish~~~~~~"<<endl;
*/

	cublasDestroy(handle);
	cusolverDnDestroy(cusolverH);

	cudaDeviceReset();


}