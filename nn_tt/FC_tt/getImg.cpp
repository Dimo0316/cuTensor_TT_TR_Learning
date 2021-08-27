// #pragma once
#include "getImg.h"

void getImg::imgTrainDataRead(const char *datapath, const char *labelpath)
{
     /***********读取图片数据***********/
    unsigned char readbuf[4];//信息数据读取空间
    FILE *f;
    f = fopen(datapath, "rb");
    fread(readbuf,1, 4, f);//读取魔数，即文件标志位
    fread(readbuf,1, 4, f);//读取数据集图像个数
    int sumOfImg = (readbuf[0] << 24) + (readbuf[1] << 16) + (readbuf[2] << 8) + readbuf[3];//图像个数
    // sumOfImg  为总计图片数量
    fread(readbuf, 1, 4, f);//读取数据集图像行数
    int imgheight = (readbuf[0] << 24) + (readbuf[1] << 16) + (readbuf[2] << 8) + readbuf[3];//图像行数  28
    fread(readbuf, 1, 4, f);//读取数据集图像列数
    int imgwidth = (readbuf[0] << 24) + (readbuf[1] << 16) + (readbuf[2] << 8) + readbuf[3];//图像列数  28

    sumOfImg = sumOfImg/BATCHSIZE;

    mImgData = new ImgData[sumOfImg];
    unsigned char *data = new unsigned char[IPNNUM*BATCHSIZE]; //28 * 28 * batchsize
    for (int i = 0; i < sumOfImg; i++)
    {
        fread(data, 1, IPNNUM*BATCHSIZE, f);//读取数据集图像列数
        for (size_t px = 0; px < IPNNUM*BATCHSIZE; px++)//图像数据归一化
        {
            mImgData[i].data[px] = data[px]/(dt)255*0.99+0.01;
        }
    }
    delete[]data;
    fclose(f);
    /***********读取标签数据***********/
    f=fopen(labelpath, "rb");
    fread(readbuf, 1, 4, f);//读取魔数，即文件标志位
    fread(readbuf, 1, 4, f);//读取数据集图像个数
    sumOfImg = (readbuf[0] << 24) + (readbuf[1] << 16) + (readbuf[2] << 8) + readbuf[3];//图像个数
    sumOfImg = sumOfImg/BATCHSIZE;
    for (int i = 0; i < sumOfImg; i++)
    {
        // for(int q=0;q<BATCHSIZE;q++)
        // {
            
        //     fread(&mImgData[i].tag[q], 1, 1, f);

        //     for (size_t j = 0; j < 10; j++)
        //     {
        //         mImgData[i].label[j+q*10] = 0.01;
        //     }
           
        //     mImgData[i].label[mImgData[i].tag[q]+q*10] = 0.99;
        // }
        for(int q=0;q<BATCHSIZE;q++)
        {
            
            fread(&mImgData[i].tag[q], 1, 1, f);

            for (size_t j = 0; j < 10; j++)
            {
                mImgData[i].label[j+q*10] = 0.0;
            }
            
            mImgData[i].label[mImgData[i].tag[q]+q*10] = 1.00;
        }

        
        /*fread(&mImgData[i].tag, 1, 1, f);//读取数据集图像列数
        for (size_t j = 0; j < 10; j++)
        {
            mImgData[i].label[j] = 0.01;
        }
        mImgData[i].label[mImgData[i].tag] = 0.99;*/

    }

    fclose(f);
}

getImg::~getImg()
{
    delete[]mImgData;
}