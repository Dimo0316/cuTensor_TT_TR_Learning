#include "head.h"


//node class,build net
class node
{
public:
    //double *value = new double[IPNNUM]; //数值，存储结点最后的状态
    double *value = NULL;
    double *W = NULL;    //结点到下一层的权值
    void initNode(int num1,int num2,int batch);//初始化函数，必须调用以初始化权值个数
    
};

void node::initNode(int num1,int num2,int batch)
{
    double *host_W = new double[num1*num2];
    cudaMalloc((void**)&value,sizeof(double)*num1*batch);
    cudaMalloc((void**)&W,sizeof(double)*num1*num2);
    srand((unsigned)time(NULL));
    for (size_t i = 0; i < num1*num2; i++)//给权值赋一个随机值
    {
        host_W[i] = rand() % 100 / double(100)*0.1;
        if (rand() % 2)
        {
            host_W[i] = -host_W[i];
        }
    }
    cudaMemcpy(W,host_W,sizeof(double)*num1*num2,cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

class net
{
public:
    node inlayer;
    node hidlayer;
    node outlayer;
    cublasHandle_t handle;
    double k1;
    double k2;//偏置
    double *Tg=NULL; //实际值（目标值）
    double *O=NULL; // 预测值
    double rate;
    net(); // 构造函数，用于初始化各层与偏置的权重
    double getloss(); // 损失函数
    void forward(double *input);//前向
    void backward(double *T);//反向
    void printresual(int trainingTime);//打印信息
};

// #define IPNNUM 784
// #define HDNNUM 256
// //#define H1 
// //#define B1
// //#define H2
// #define OPNNUM 10
// #define BATCHSIZE 64

net::net() {
    //初始化输入与隐藏偏置权重
    srand((unsigned)time(NULL));
    k1= (rand() % 100) / (double)100;
    k2 = (rand() % 100) / (double)100;
    rate = 0.025; //学习率
    O = new double[OPNNUM*BATCHSIZE];
    //初始化输入到隐藏节点个数
    //init matrix W1 IPNNUM*HDNNUM
    inlayer.initNode(IPNNUM,HDNNUM,BATCHSIZE);
    hidlayer.initNode(HDNNUM,OPNNUM,BATCHSIZE);
    outlayer.initNode(OPNNUM,OPNNUM,BATCHSIZE);
    cudaMalloc((void**)&Tg,sizeof(double)*OPNNUM*BATCHSIZE);
    cublasCreate(&handle);

}

//损失函数
double net::getloss() {
    double mloss = 0;
    mloss = loss_gpu(Tg,outlayer.value,OPNNUM,BATCHSIZE,handle);  
    return mloss;
}
//前向
void net::forward(double *input) {

    cudaMemcpy(inlayer.value,input,sizeof(double)*IPNNUM*BATCHSIZE,cudaMemcpyHostToDevice);
    //hidlayer.value = inlayer.W * inlayer.value;
    // 激活函数
    //outlayer,value = hidlayer.W * hidlayer,value;
    forward_cuda(inlayer.value,inlayer.W,hidlayer.value,hidlayer.W,outlayer.value,k1,k2,IPNNUM,HDNNUM,OPNNUM,BATCHSIZE,handle);
}
void net::backward(double *T) {
    cudaMemcpy(Tg,T,sizeof(double)*OPNNUM*BATCHSIZE,cudaMemcpyHostToDevice);
    back_cuda(Tg,outlayer.value,hidlayer.value,hidlayer.W,inlayer.value,inlayer.W,IPNNUM,HDNNUM,OPNNUM,BATCHSIZE,rate,handle);
}

void net::printresual(int trainingTimes)
{
    double loss = getloss();
    cout << "train times:" << trainingTimes << endl;
    cout << "loss:" << loss << endl;
    
}

class ImgData//单张图像
{
public:
    unsigned char tag[BATCHSIZE];
    double data[IPNNUM*BATCHSIZE];
    double label[OPNNUM*BATCHSIZE];
};

class getImg {
public:
    ImgData* mImgData;
    void imgTrainDataRead(const char *datapath, const char *labelpath);
    ~getImg();
};

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
            mImgData[i].data[px] = data[px]/(double)255*0.99+0.01;
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
        for(int q=0;q<BATCHSIZE;q++)
        {
            
            fread(&mImgData[i].tag[q], 1, 1, f);

            for (size_t j = 0; j < 10; j++)
            {
                mImgData[i].label[j+q*10] = 0.01;
            }
           
            mImgData[i].label[mImgData[i].tag[q]+q*10] = 0.99;
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
void AccuracyRate(int time, net *mnet, getImg *mImg)//精确率评估
{
    double tagright = 0;//正确个数统计
    for (size_t count = 0; count < 10000/BATCHSIZE; count++)
    {
        mnet->forward(mImg->mImgData[count].data);//前向传播
        double value;
        int gettag;
        cudaMemcpy(mnet->O,mnet->outlayer.value,sizeof(double)*OPNNUM*BATCHSIZE,cudaMemcpyDeviceToHost);
        
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
    cout << "zheng que lv:" << tagright / 10000 << endl;
}
int main() {
    clock_t t1,t2;
    double times=0.0;
    getImg mGetTrainImg;
    mGetTrainImg.imgTrainDataRead("../train-images.idx3-ubyte", "../train-labels.idx1-ubyte");
    getImg mGetTestImg;
    mGetTestImg.imgTrainDataRead("../t10k-images.idx3-ubyte", "../t10k-labels.idx1-ubyte");   
    warmup();
    t1=clock(); 
    net mnet;
    for(int j=0;j<20;j++)
    {
    for (int i = 0; i < 60000/BATCHSIZE; ++i) {
        mnet.forward(mGetTrainImg.mImgData[i].data);
        mnet.backward(mGetTrainImg.mImgData[i].label);
        if (i % 100 == 0)
            mnet.printresual(i);//信息打印
    }
    AccuracyRate(j, &mnet, &mGetTestImg);
    mnet.rate = mnet.rate-mnet.rate*0.001;
}
    t2=clock();
    times = (double)(t2-t1)/CLOCKS_PER_SEC;    
    cout<<"cost time :"<<times<<"s"<<endl;
    return 0;
}

// 10个epoch，精度96.26%， cost time is 3.04s


