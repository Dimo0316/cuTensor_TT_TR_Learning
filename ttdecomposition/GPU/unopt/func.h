/*************************************************************************
	> File Name: func.h
	> Author: honghao
	> Mail: honghao@shu.edu.cn 
	> Created Time: 2020年04月2日 星期四 14时25分17秒
 ************************************************************************/
#ifndef GUARD_func_h
#define GUARD_func_h

typedef float dt;

void warmupcu();
void printMatrix(int m, int n, const dt*A, int lda, const char* name);
void printMatrix_Device(int m, int n, const dt*d_A, int lda, const char* name);
float ttdec_gpu_float_unopt(float* h_tensor, int* h_n_array, int* h_r_array, double eps);
void genTTTensor(dt *T,long a,long b,long c,long r);
__global__ void sub(dt *A,dt *B,long a,long b,long c);

#endif
