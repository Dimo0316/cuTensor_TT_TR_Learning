#include "head.h"
 
int main()
{
 
  int chi = 16;
    
  int numsweep = 5;
  int dispon = 2; 
  bool updateon = true;
  int maxit=2;
  int krydim=4;
  srand(time(0));
 
  int chid = 2;
dt *sP,*sM,*sI;
  cudaHostAlloc((void**)&sP,sizeof(dt)*2*2,0);
  cudaHostAlloc((void**)&sM,sizeof(dt)*2*2,0);
  cudaHostAlloc((void**)&sI,sizeof(dt)*2*2,0);
  sP[0] = 0;sP[1] = sqrt(2);sP[2] = 0;sP[3] = 0;
  sM[0] = 0;sM[1] = 0;sM[2] = sqrt(2);sM[3] = 0;
  sI[0] = 1;sI[1] = 0;sI[2] = 0;sI[3] = 1;

  //M 为一个 2*2*4*4 的tensor
  dt *M;
  int k;
  cudaHostAlloc((void**)&M,sizeof(dt)*2*2*4*4,0);
  for(k=0;k<2*2*4*4;k++)
  {
    if(k==0)
      M[k]=1;
    else if(k==5)
      M[k]=sqrt(2);
    else if(k==9)
      M[k] = 1;
    else if(k==10)
      M[k] = sqrt(2);
    else if(k==23)
      M[k] = sqrt(2);
    else if(k==46)
      M[k] = sqrt(2);
    else if(k==54)
      M[k] = 1;
    else if(k==63)
      M[k]=1;
    else
      M[k]=0;
  }

  //ML,MR为边界
  dt *ML,*MR;
  cudaHostAlloc((void**)&ML,sizeof(dt)*4*1*1,0);
  cudaHostAlloc((void**)&MR,sizeof(dt)*4*1*1,0);
  ML[0] = 1;ML[1] = 0;ML[2] = 0;ML[3] = 0;
  MR[0] = 0;MR[1] = 0;MR[2] = 0;MR[3] = 1;
  for(int hh=200;hh<203;hh=hh+200)
  {
    int Nsites =hh;  //4个MPS 、MPO tensor
    cout<<"--"<<hh<<"--"<<endl;

  //A1 - A4 为MPS的4个tensor
  /*
  *   A1 size:1*2*2
  *   A2 size:2*2*4
  *   A3 size:4*2*2
  *   A4 size:2*2*1
  */
  dt *A[Nsites];
  //A 的维度顺序为 2 0 1 -> sizeA[0] sizeA[1] sizeA[2]
  dt *d_A[Nsites];
  int sizeA[Nsites][3];
  sizeA[0][0]=1;
  sizeA[0][1]=2;
  sizeA[0][2]=2;

  float temp;
  for(int i=1;i<Nsites;i++)
  {
    if(chi < sizeA[i-1][1]*chid)
    {
      temp = chi;
    }else
    {
      temp =sizeA[i-1][1]*chid;
    }
    if(pow(chid,(Nsites-i-1)) < temp)
    {
      temp = pow(chid,(Nsites-i-1));
    }

    sizeA[i][0]=sizeA[i-1][2];
    sizeA[i][1]=chid;
    sizeA[i][2]=temp;
  }
  for(int i=0;i<Nsites;i++)
  {
    cudaHostAlloc((void**)&A[i],sizeof(float)*sizeA[i][0]*sizeA[i][1]*sizeA[i][2],0);
    cudaMalloc((void**)&d_A[i],sizeof(float)*sizeA[i][0]*sizeA[i][1]*sizeA[i][2]);
    for(int k=0;k<sizeA[i][0]*sizeA[i][1]*sizeA[i][2];k++)
    {
      A[i][k] = rand()*0.1/(RAND_MAX*0.1);
    }
    cudaMemcpy(d_A[i],A[i],sizeof(dt)*sizeA[i][0]*sizeA[i][1]*sizeA[i][2],cudaMemcpyHostToDevice);

  }

  
  doDMRG_MPO(M,d_A,ML,MR,sizeA,numsweep,dispon,updateon,maxit,krydim,hh);
}
  cudaFreeHost(M);
  cudaFreeHost(sP);
  cudaFreeHost(sI);
  cudaFreeHost(sM);
}