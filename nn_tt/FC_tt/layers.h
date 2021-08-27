#ifndef _LAYERS_H_
#define _LAYERS_H_
#include "head.h"

class layerFull
{
    public:
        dt *value = NULL;
        dt *W = NULL;    //结点到前一层的权值
        void initLayer(int num1,int num2,int batch);//初始化函数，必须调用以初始化权值个数
        void malloc_W(int num1,int num2,int batch);
        ~layerFull();
};


class layerTTG
{
    public:
        dt *value = NULL;
        dt *G = NULL;    //结点到前一层的权值
        void initLayer(int out_mode, int rank2, int rank1, int inp_mode, int out_value2, int batch);//初始化函数，必须调用以初始化权值个数
        void malloc_G(int out_mode, int rank2, int rank1, int inp_mode, int out_value2, int batch);
        ~layerTTG();
};

#endif