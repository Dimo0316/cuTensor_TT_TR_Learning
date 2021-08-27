// #pragma once
#include "layers.h"

void layerFull::initLayer(int num1,int num2,int batch)
{
    dt *host_W = new dt[num1*num2];
    cudaMalloc((void**)&value,sizeof(dt)*num1*batch);
    cudaMalloc((void**)&W,sizeof(dt)*num1*num2);
    srand((unsigned)time(NULL));
    for (size_t i = 0; i < num1*num2; i++)//给权值赋一个随机值
    {
        host_W[i] = rand() % 100 / dt(100)*0.1;
        if (rand() % 2)
        {
            host_W[i] = -host_W[i];
        }
    }
    cudaMemcpy(W,host_W,sizeof(dt)*num1*num2,cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

void layerFull::malloc_W(int num1,int num2,int batch)
{
    cudaMalloc((void**)&value,sizeof(dt)*num1*batch);
    cudaMalloc((void**)&W,sizeof(dt)*num1*num2);
}

layerFull::~layerFull(){
    // printf("This is invoke of layerFull\n");
    if(value) cudaFree(value); value = NULL;
    if(W) cudaFree(W); W = NULL;
}

void layerTTG::initLayer(int out_mode, int rank2, int rank1, int inp_mode, int out_value2, int batch)
{
    dt *host_W = NULL;
    int number_W = out_mode*rank2*rank1*inp_mode;
    int number_Value = out_mode*rank2*out_value2*batch;
    cudaHostAlloc((void**)&host_W,sizeof(dt)*number_W,0);
    cudaMalloc((void**)&value,sizeof(dt)*number_Value);
    cudaMalloc((void**)&G,sizeof(dt)*number_W);
    srand((unsigned)time(NULL));
    for (size_t i = 0; i < number_W; i++)//给权值赋一个随机值
    {
        host_W[i] = rand() % 100 / dt(100)*0.1;
        if (rand() % 2)
        {
            host_W[i] = -host_W[i];
        }
    }
    cudaMemcpy(G,host_W,sizeof(dt)*number_W,cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    if(host_W) cudaFreeHost(host_W); host_W = NULL;
}

void layerTTG::malloc_G(int out_mode, int rank2, int rank1, int inp_mode, int out_value2, int batch)
{
    cudaMalloc((void**)&value,sizeof(dt)*out_mode*rank2*out_value2*batch);
    cudaMalloc((void**)&G,sizeof(dt)*out_mode*rank2*rank1*inp_mode);
}

layerTTG::~layerTTG(){
    // printf("This is invoke of layerTTG\n");
    if(value) cudaFree(value); value = NULL;
    if(G) cudaFree(G); G = NULL;
}