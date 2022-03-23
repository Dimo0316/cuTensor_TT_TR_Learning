/*************************************************************************
	> File Name: func.h
	> Author: honghao
	> Mail: honghao@shu.edu.cn 
	> Created Time: 2020年04月2日 星期四 14时25分17秒
 ************************************************************************/
#include "head.h"


//进行3维的 稠密  张量分解
int main(){
ofstream ofs;
ofs.open("result_half.txt", ios::out);
for(int i=1700; i<1900;i=i+100){
	warmupcu();
	// for(int i = 0; i < d1*d2*d3 ;++i)
	// 	cout << h_tensor[i] << "  " ;
	// cout<<endl;
	double duration = 0.0;
	// ttdec_gpu_float(h_tensor, h_n_array, h_r_array, eps);
	float time = 0.0;
	int calculateTimes = 1;
	// 	
	dt *h_tensor;
	int q = i, r = 50;
	int d1 = q, d2 = q, d3 = q;
	cout<<d1<<endl;
	int p = 0;
	// const int k = 8*7; //56
 //    const int s = 8;
 //    const int p = 2;
 //    const int ks = k+s;
    // float time = 0;
	int r_mul8 = 56;//56  //Multiple of 8
	if(i > 500 && q>100){
		r_mul8 = q/10/8*8;
	}
	// r_mul8 = 2;
	printf("r_mul8: %d\n", r_mul8);
	// cout << "r_mul8: "<< r_mul8 << endl;
	cudaHostAlloc((void**)&h_tensor,sizeof(dt)*d1*d2*d3,0);

	genTTTensor(h_tensor,d1,d2,d3,r);
	int h_n_array[3] = {d1, d2, d3};
	int h_r_array[4] = {1, r_mul8, r_mul8, 1};
	double eps[2] = {1.e-6,1.e-8};
	float singleError[1]={0.0f}, singleComRatio[1] = {0.0f};
	if(calculateTimes != 1)
		ttdec_gpu_float(h_tensor, h_n_array, h_r_array, eps, singleError, singleComRatio);
	for(int i = 0; i < calculateTimes; ++i){
		cout << "---------------"<< i+1 << "-------------------" << endl;
		// clock_t  start, end;
		// start = clock();
		time = ttdec_gpu_float(h_tensor, h_n_array, h_r_array, eps, singleError, singleComRatio);
		// end = clock();
		// time = (double)(end - start) / CLOCKS_PER_SEC;
		// cout <<"\n************************************\n" << "run time: "<< time << "\n*********************************************************\n----------------------------------------------" << endl;
		duration += time;
	}
	
    printf( "\n******************************\nTime: %f s\n******************************\n", duration/(float)calculateTimes ); 
	ofs << duration/(float)calculateTimes<<endl;

	if(h_tensor) cudaFreeHost(h_tensor); h_tensor = NULL;
	
}
ofs.close();
return 0;
}
