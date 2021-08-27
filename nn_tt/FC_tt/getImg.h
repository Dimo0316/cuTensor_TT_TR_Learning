#ifndef _GETIMG_H_
#define _GETIMG_H_
#include "head.h"

class ImgData//单张图像*Batchsize
{
public:
    unsigned char tag[BATCHSIZE];
    dt data[IPNNUM*BATCHSIZE];
    dt label[OPNNUM*BATCHSIZE];
};

class getImg {
public:
    ImgData* mImgData;
    void imgTrainDataRead(const char *datapath, const char *labelpath);
    ~getImg();
};

#endif