// #pragma once
#include "net.h"

net::net() {
    //初始化输入与隐藏偏置权重
    srand((unsigned)time(NULL));
    // b1 = 0.0;
    // b2 = 0.0;
    b2 = (rand() % 100) / (dt)100;
    b1= (rand() % 100) / (dt)100;
    rate = 0.025; //学习率
    O = new dt[OPNNUM*BATCHSIZE];
    //初始化输入到隐藏节点个数
    //init matrix W1 IPNNUM*HDNNUM
   
    // inlayer.init_W(IPNNUM,K2,BATCHSIZE);
    // hidlayer_U2.init_W(K2,K1,BATCHSIZE);
    // hidlayer_B1.init_W(K1,HDNNUM,BATCHSIZE);
    // inlayer_tmp.initLayer(IPNNUM,HDNNUM,BATCHSIZE); 
    // cublasCreate(&handle);
    // ht(inlayer_tmp.W,inlayer.W,hidlayer_B1.W,hidlayer_U2.W,HDNNUM,IPNNUM,K1,K2,handle);
    // hidlayer_U1.initLayer(HDNNUM,OPNNUM,BATCHSIZE);
    // //hidlayer.initLayer(HDNNUM,OPNNUM,BATCHSIZE);
    // outlayer.initLayer(OPNNUM,OPNNUM,BATCHSIZE);
    // cudaMalloc((void**)&Tg,sizeof(dt)*OPNNUM*BATCHSIZE);
    out_values1[0] = IPNNUM/(ranks_1[0]*inp_modes_1[0]);
    out_values1[1] = out_values1[0]*out_modes_1[0]/(inp_modes_1[1]);
    out_values1[2] = out_values1[1]*out_modes_1[1]/(inp_modes_1[2]);
    out_values1[3] = out_values1[2]*out_modes_1[2]/(inp_modes_1[3]);
    
    // for(int i=0;i<4;i++){
    //     printf("%d\n", out_values1[i]);
    // }
    inlayer.initLayer(IPNNUM,IPNNUM,BATCHSIZE);
    hidlayer_TT1_G1.initLayer(out_modes_1[0], ranks_1[1], ranks_1[0], inp_modes_1[0], out_values1[0], BATCHSIZE);
    hidlayer_TT1_G2.initLayer(out_modes_1[1], ranks_1[2], ranks_1[1], inp_modes_1[1], out_values1[1], BATCHSIZE);
    hidlayer_TT1_G3.initLayer(out_modes_1[2], ranks_1[3], ranks_1[2], inp_modes_1[2], out_values1[2], BATCHSIZE);
    hidlayer_TT1_G4.initLayer(out_modes_1[3], ranks_1[4], ranks_1[3], inp_modes_1[3], out_values1[3], BATCHSIZE);
    outlayer.initLayer(OPNNUM,HDNNUM,BATCHSIZE);
    cudaMalloc((void**)&Tg,sizeof(dt)*OPNNUM*BATCHSIZE);
    cublasCreate(&handle);
///TODO ttDecomposition
}


net::~net(){
	// printf("This is invoke of net\n");
	if(handle) cublasDestroy(handle); handle = NULL;
	if(Tg) cudaFree(Tg); Tg = NULL;
	delete O;
}

//损失函数
dt net::getloss() {
    dt mloss = 0;
    mloss = loss_gpu_cross(Tg,outlayer.value,OPNNUM,BATCHSIZE,handle);  
    return mloss;
}


//前向
void net::forward(dt *input) {
    cudaMemcpy(inlayer.value,input,sizeof(dt)*IPNNUM*BATCHSIZE,cudaMemcpyHostToDevice);

    forward_cuda(inlayer.value, hidlayer_TT1_G1.G, 
    			 hidlayer_TT1_G1.value, hidlayer_TT1_G2.G,
                 hidlayer_TT1_G2.value, hidlayer_TT1_G3.G,
                 hidlayer_TT1_G3.value, hidlayer_TT1_G4.G,
                 hidlayer_TT1_G4.value, outlayer.W,
                 outlayer.value, b1,b2, 
                 inp_modes_1, out_modes_1, ranks_1, out_values1, OPNNUM, BATCHSIZE, handle);
    
}

//反向
void net::backward(dt *T) {
    cudaMemcpy(Tg, T, sizeof(dt)*OPNNUM*BATCHSIZE,cudaMemcpyHostToDevice);    
///TODO
    back_cuda(Tg, outlayer.value, outlayer.W, 
    			hidlayer_TT1_G4.value, hidlayer_TT1_G4.G, hidlayer_TT1_G3.value, hidlayer_TT1_G3.G, 
    			hidlayer_TT1_G2.value, hidlayer_TT1_G2.G, hidlayer_TT1_G1.value, hidlayer_TT1_G1.G,
    			inlayer.value,
    			inp_modes_1, out_modes_1, ranks_1, out_values1, OPNNUM, BATCHSIZE, rate, handle);
}

void net::printresual(int trainingTimes)
{
    dt loss = getloss();
    cout << "train times:" << trainingTimes << endl;
    cout << "loss:" << loss << endl;
}

