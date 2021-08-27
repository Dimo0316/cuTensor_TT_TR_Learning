#ifndef _NET_H_
#define _NET_H_
#include "head.h"
#include "layers.h"
class net
{
public:
    layerFull inlayer;
    layerTTG hidlayer_TT1_G1;
    layerTTG hidlayer_TT1_G2;
    layerTTG hidlayer_TT1_G3;
    layerTTG hidlayer_TT1_G4;
    layerFull outlayer;
    cublasHandle_t handle;

    int inp_modes_1[4] = {4,7,4,7};
    int out_modes_1[4] = {4,8,4,8};
    int ranks_1[5] = {1,4,4,4,1};
    int out_values1[4] = {1, 1, 1, 1};
    
    
    dt b1; 
    dt b2;//偏置
    dt *Tg=NULL; //实际值（目标值）
    dt *O=NULL; // 预测值
    dt rate = 0.01;
    dt momentum = 0.9;
    net(); // 构造函数，用于初始化各层与偏置的权重
    ~net(); 
    dt getloss(); // 损失函数
    void forward(dt *input); //前向
    void backward(dt *T); //反向
    void printresual(int trainingTime); //打印信息
};

#endif