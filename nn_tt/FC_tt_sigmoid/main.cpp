// #pragma once
#include "head.h"
#include "getImg.h"
#include "layers.h"
#include "net.h"

void AccuracyRate(int time, net *mnet, getImg *mImg)//精确率评估
{
    dt tagright = 0;//正确个数统计
    for (size_t count = 0; count < 10000/BATCHSIZE; count++)
    {
        mnet->forward(mImg->mImgData[count].data);//前向传播
        dt value;
        int gettag;
        cudaMemcpy(mnet->O,mnet->outlayer.value,sizeof(dt)*OPNNUM*BATCHSIZE,cudaMemcpyDeviceToHost);
        
        for(int i = 0; i < BATCHSIZE; ++i) {
            value = -100;gettag = -100;
            for (size_t j = 0; j < 10; j++)
            {
                if (mnet->O[j+i*10] > value)
                {
                    value = mnet->O[j+i*10];
                    gettag = j;
                }
            }
            if (mImg->mImgData[count].tag[i] == gettag)
            {
                tagright++;
            }
        }
    }
    //mnet.printresual(0);//信息打印
    cout << "num." << time + 1 << ":  ";
    cout << "accuracy:" << tagright / 10000 << endl;
}

int main() {
    clock_t t1,t2;
    dt times=0.0;
    getImg mGetTrainImg;
    //mGetTrainImg.imgTrainDataRead("D:\\IIP\\TensorNet\\decomposition\\Test_fully_1207\\mnist\\train-images.idx3-ubyte", "D:\\IIP\\TensorNet\\decomposition\\Test_fully_1207\\mnist\\train-labels.idx1-ubyte");
    mGetTrainImg.imgTrainDataRead("../train-images.idx3-ubyte", "../train-labels.idx1-ubyte");
    getImg mGetTestImg;
    //mGetTestImg.imgTrainDataRead("D:\\IIP\\TensorNet\\decomposition\\Test_fully_1207\\mnist\\t10k-images.idx3-ubyte", "D:\\IIP\\TensorNet\\decomposition\\Test_fully_1207\\mnist\\t10k-labels.idx1-ubyte");
    mGetTestImg.imgTrainDataRead("../t10k-images.idx3-ubyte", "../t10k-labels.idx1-ubyte");
    net mnet;
    int epoch = 10;
    warmup();   
    for(int j=0;j<epoch;j++){
        for (int i = 0; i < 600; ++i) {
            t1=clock(); 
            mnet.forward(mGetTrainImg.mImgData[i].data);
            mnet.backward(mGetTrainImg.mImgData[i].label);
            t2=clock();
            times += (dt)(t2-t1)/CLOCKS_PER_SEC;
            if (i % 300 == 0){
                mnet.printresual(i);//信息打印
                // AccuracyRate(j, &mnet, &mGetTestImg);
            }
        }
        AccuracyRate(j, &mnet, &mGetTestImg);
        mnet.rate = mnet.rate-mnet.rate*0.1;
    }
    
    cout<<"each epoch cost time :"<<times / (dt)epoch<<"s"<<endl;
    return 0;
}

// 10个epoch，精度96.26%， cost time is 3.26s

// 10个epoch，精度95.67%  cost time is 3.24s

