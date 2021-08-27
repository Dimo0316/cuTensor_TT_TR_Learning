#include <stdlib.h>
#include <stdio.h>
#include<iostream>
#include<fstream>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <cutensor.h>
using namespace std;
using std::vector;
using std::unordered_map;


void ncon_1(float *A_d,float *B_d,float *C_d,vector<int> modeA,vector<int> modeB,vector<int> modeC,unordered_map<int, int64_t> extent,cutensorHandle_t handle)
{
    typedef float floatTypeA;
    typedef float floatTypeB;
    typedef float floatTypeC;
    typedef float floatTypeCompute;

    cudaDataType_t typeA = CUDA_R_32F;
    cudaDataType_t typeB = CUDA_R_32F;
    cudaDataType_t typeC = CUDA_R_32F;
    cutensorComputeType_t typeCompute = CUTENSOR_R_MIN_32F;

    floatTypeCompute alpha = (floatTypeCompute)1.1f;
    floatTypeCompute beta  = (floatTypeCompute)0.0f;


    int nmodeA = modeA.size();
    int nmodeB = modeB.size();
    int nmodeC = modeC.size();

    cout<<nmodeA<<endl;



    vector<int64_t> extentC;
    for (auto mode : modeC)
        extentC.push_back(extent[mode]);
    vector<int64_t> extentA;
    for (auto mode : modeA)
        extentA.push_back(extent[mode]);
    vector<int64_t> extentB;
    for (auto mode : modeB)
        extentB.push_back(extent[mode]);

    size_t elementsA = 1;
    for (auto mode : modeA)
        elementsA *= extent[mode];
    size_t elementsB = 1;
    for (auto mode : modeB)
        elementsB *= extent[mode];
    size_t elementsC = 1;
    for (auto mode : modeC)
        elementsC *= extent[mode];

    cutensorTensorDescriptor_t descA;
    cutensorInitTensorDescriptor(&handle,
                 &descA,
                 nmodeA,
                 extentA.data(),
                 NULL,/*stride*/
                 typeA, CUTENSOR_OP_IDENTITY);

    cutensorTensorDescriptor_t descB;
   cutensorInitTensorDescriptor(&handle,
                 &descB,
                 nmodeB,
                 extentB.data(),
                 NULL,/*stride*/
                 typeB, CUTENSOR_OP_IDENTITY);

    cutensorTensorDescriptor_t descC;
   cutensorInitTensorDescriptor( &handle,
                 &descC,
                 nmodeC,
                 extentC.data(),
                 NULL,/*stride*/
                 typeC, CUTENSOR_OP_IDENTITY);

     uint32_t alignmentRequirementA;
     cutensorGetAlignmentRequirement(&handle,
                  A_d,
                  &descA,
                  &alignmentRequirementA);

     uint32_t alignmentRequirementB;
     cutensorGetAlignmentRequirement(&handle,
                  B_d,
                  &descB,
                  &alignmentRequirementB);

     uint32_t alignmentRequirementC;
     cutensorGetAlignmentRequirement(&handle,
                  C_d,
                  &descC, 
                  &alignmentRequirementC);

    cutensorContractionDescriptor_t desc;
   cutensorInitContractionDescriptor(&handle, 
                 &desc,
                 &descA, modeA.data(), alignmentRequirementA,
                 &descB, modeB.data(), alignmentRequirementB,
                 &descC, modeC.data(), alignmentRequirementC,
                 &descC, modeC.data(), alignmentRequirementC,
                 typeCompute);

    cutensorContractionFind_t find;
    cutensorInitContractionFind( 
                 &handle, &find, 
                 CUTENSOR_ALGO_DEFAULT);

    uint64_t worksize = 0;
    cutensorContractionGetWorkspace(&handle,
                 &desc,
                 &find,
                 CUTENSOR_WORKSPACE_RECOMMENDED, &worksize);

    void *work = nullptr;
    if (worksize > 0)
    {
        if (cudaSuccess != cudaMalloc(&work, worksize))
        {
            work = nullptr;
            worksize = 0;
        }
    } 
    /**************************
     * Create Contraction Plan
     **************************/
    cutensorContractionPlan_t plan;
    cutensorInitContractionPlan(&handle,
                 &plan,
                 &desc,
                 &find,
                 worksize);

    /**********************
     * Run
     **********************/
    cutensorStatus_t err;
  

        err = cutensorContraction(&handle,
                                  &plan,
                                  (void*) &alpha, A_d, B_d,
                                  (void*) &beta,  C_d, C_d, 
                                  work, worksize, 0 /* stream */);


        if (err != CUTENSOR_STATUS_SUCCESS)
        {
            cout<<"over"<<endl; 
            printf("ERROR: %s in line %d\n", cutensorGetErrorString(err), __LINE__);
        }

    /*************************/


    if (A_d) cudaFree(A_d);
    if (B_d) cudaFree(B_d);
    if (work) cudaFree(work);

}

int main()
{
    int n = 2;
    int test;
    float *L = new float[n*n*n];
    float *M = new float[n*n*n*n];
    float *A = new float[n*n*n];
    

    for (int i = 0; i < n*n*n; i++)
        L[i] = (((float) rand())/RAND_MAX - 0.5)*100;
    for (int i = 0; i < n*n*n*n; i++)
        M[i] = (((float) rand())/RAND_MAX - 0.5)*100;
    for (int i = 0; i < n*n*n; i++)
        A[i] = (((float) rand())/RAND_MAX - 0.5)*100;


    float *L_d,*M_d,*A_d;
    cudaMalloc((void**)&L_d,sizeof(float)*n*n*n);
    cudaMalloc((void**)&M_d,sizeof(float)*n*n*n*n);
    cudaMalloc((void**)&A_d,sizeof(float)*n*n*n);


    cudaMemcpy(L_d,L,sizeof(float)*n*n*n,cudaMemcpyHostToDevice);
    cudaMemcpy(M_d,M,sizeof(float)*n*n*n,cudaMemcpyHostToDevice);
    cudaMemcpy(A_d,A,sizeof(float)*n*n*n,cudaMemcpyHostToDevice);


    float *LM_d,*ALM_d,*out_d;
    cudaMalloc((void**)&LM_d,sizeof(float)*n*n*n*n*n);
    cudaMalloc((void**)&ALM_d,sizeof(float)*n*n*n*n);
    cudaMalloc((void**)&out_d,sizeof(float)*n*n*n);
    cutensorHandle_t handle;
    cutensorInit(&handle);

    vector<int> modeLM{'a','b','d','e','f'};
    vector<int> modeL{'a','b','c'};
    vector<int> modeM{'d','e','f','c'};
    unordered_map<int, int64_t> extent;
    extent['a'] = n;
    extent['b'] = n;
    extent['c'] = n;
    extent['d'] = n;
    extent['e'] = n;
    extent['f'] = n;

    ncon_1(L_d,M_d,LM_d,modeL,modeM,modeLM,extent,handle);
    cout<<"++++++++"<<endl;

    float *LM = new float[n*n*n*n*n];
    cudaMemcpy(LM,LM_d,sizeof(float)*n*n*n*n*n,cudaMemcpyDeviceToHost);

    for(int i=0;i<n*n*n*n*n;i++)
    {
        printf("%.2f _",LM[i]);
    }
    printf("\n");

   return 0;
}