#include "head.h"

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
/*
	dt *p_temp3 = new dt[size_psi];
	cudaMemcpy(p_temp3,d_psiGround,sizeof(dt)*size_psi,cudaMemcpyDeviceToHost);
	cout<<"--lanczos suan fa kai shi zhi qian de psi :"<<endl;
	for(int i=0;i<size_psi;i++)
	{
		cout<<p_temp3[i]<<" _ ";
	}
	cout<<endl;
*/
	cudaMalloc((void**)&d_A,sizeof(dt)*krydim*krydim); 
	cudaMalloc((void**)&d_W,sizeof(dt)*krydim);
	cudaMalloc((void**)&d_psi_temp,sizeof(dt)*size_psi*krydim);
	cudaMalloc((void**)&d_ig_temp,sizeof(dt)*size_psi);

	// 循环 maxit 次

	dt sh=0.0;

    float alpha = 1.0;
    float beta = 0.0;
    //float beta2 = -1.0;

    for(int j=0;j<1;j++)
    {
		sh=norm(d_psiGround,sh,size_psi,handle);
		if(sh < 1e-16)
		{
			sh = 1e-16;
		}	
		sh=1/sh;
		division(d_psiGround,sh,d_psiGround,size_psi,1,1,handle);
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

	    dt *d_LMR;
		cudaMalloc((void**)&d_LMR,sizeof(dt)*sizeL[0]*sizeL[1]*2*2*sizeR[0]*sizeR[1]);
		
	    vector<int> modeR{'g','h','f'};
	    vector<int> modeLMR{'b','d','h','a','e','g'};
	    unordered_map<int, int64_t> extent2;
	    extent2['a'] = sizeL[0];
	    extent2['b'] = sizeL[1];	    
	    extent2['d'] = 2;
	    extent2['e'] = 2;
	    extent2['g'] = sizeR[0];
	    extent2['h'] = sizeR[1];

	    ncon_1(d_LM,d_R,d_LMR,modeLM,modeR,modeLMR,extent2,tensor_handle); //0 多，但是也有非0
/*
	    dt *p_temp4 = new dt[sizeL[0]*sizeL[1]*2*2*sizeR[0]*sizeR[1]];
			cudaMemcpy(p_temp4,d_LMR,sizeof(dt)*sizeL[0]*sizeL[1]*2*2*sizeR[0]*sizeR[1],cudaMemcpyDeviceToHost);
			cout<<"mei ci de LRM"<<endl;
			for(int i=0;i<sizeL[0]*sizeL[1]*2*2*sizeR[0]*sizeR[1];i++)
			{
				cout<<p_temp4[i]<<" _ ";
			}
			cout<<endl;
*/

	 
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

/*	  		dt *p_temp = new dt[size_psi];
			cudaMemcpy(p_temp,p[ip],sizeof(dt)*size_psi,cudaMemcpyDeviceToHost);
			cout<<"*****p[ip] value is*****"<<endl;
			for(int i=0;i<size_psi;i++)
			{
				cout<<p_temp[i]<<" _ ";
			}
			cout<<endl;*/
	  			  		
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
	  			
	  			//cout<<ip<<"cuda_dot value is :"<<temp<<endl;
/*
				cudaMemcpy(p_temp,p[ig],sizeof(dt)*size_psi,cudaMemcpyDeviceToHost);
	  			cout<<"ip:"<<ip<<", ig:"<<ig<<":p[ig] value is(before cheng cuda_dot):"<<endl;
				for(int i=0;i<size_psi;i++)
				{
					cout<<p_temp[i]<<" _ ";
				}
				cout<<endl;
*/
	  			//division(p[ig],temp,d_ig_temp,size_psi,1,1,handle);
	  			//cudaDeviceSynchronize();
	  			dt beta2 = -temp;
	  			/*
				cudaMemcpy(p_temp,p[ip],sizeof(dt)*size_psi,cudaMemcpyDeviceToHost);
				cout<<"ip:"<<ip<<", ig:"<<ig<<":p[ip] value:(yu ig xiang  jian zhi qian)"<<endl;
				for(int i=0;i<size_psi;i++)
				{
					cout<<p_temp[i]<<" _ ";
				}
				cout<<endl;
*/
	  			cublasSgeam(handle,CUBLAS_OP_N,CUBLAS_OP_N,
	  			            size_psi,1,
	  			            &alpha,p[ip],size_psi,
	  			            &beta2,p[ig],size_psi,p[ip],size_psi
	  			            );
	  		/*	
				cudaMemcpy(p_temp,p[ip],sizeof(dt)*size_psi,cudaMemcpyDeviceToHost);
				cout<<"ip:"<<ip<<", ig:"<<ig<<":p[ip] value is(after update):"<<endl;
				for(int i=0;i<size_psi;i++)
				{
					cout<<p_temp[i]<<" _ ";
				}
				cout<<endl;
				*/
	  			cudaDeviceSynchronize();
	  			sh = norm(p[ip],sh,size_psi,handle);
	  			//cout<<"ip:"<<ip<<", ig:"<<ig<<":p[ip] zheng jiao :"<<sh<<endl;

	  			if(sh < 1e-16)
	  			{
	  				sh = 1e-16;
	  			}	  			
	  			sh = 1/sh;
	  			division(p[ip],sh,p[ip],size_psi,1,1,handle);	
/*
	  			cudaMemcpy(p_temp,p[ip],sizeof(dt)*size_psi,cudaMemcpyDeviceToHost);
				cout<<"ip:"<<ip<<", ig:"<<ig<<":p[ip] value is(zheng  ze  hua  zhi  hou):"<<endl;
				for(int i=0;i<size_psi;i++)
				{
					cout<<p_temp[i]<<" _ ";
				}
				cout<<endl;

	  			cout<<"+++++++++++"<<endl;
*/	  				  		  			
	  		}
/*	  		
	  		dt *p_temp2 = new dt[size_psi];
			cudaMemcpy(p_temp2,p[ip],sizeof(dt)*size_psi,cudaMemcpyDeviceToHost);
			cout<<ip<<":*****p[ip] value is(after update)*****"<<endl;
			for(int i=0;i<size_psi;i++)
			{
				cout<<p_temp2[i]<<" _ ";
			}
			cout<<endl;	 	*/
	  	}
	  	//cudaFree(d_LM);
	  	//cudaFree(d_LMR);
	  	
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
	if(sh < 1e-16)
	{
		sh = 1e-16;
	}	
	division(d_psiGround,sh,d_psiGround,size_psi,1,1,handle);
	cudaDeviceSynchronize();  

	dt w=0.0;
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