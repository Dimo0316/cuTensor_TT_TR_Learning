/*************************************************************************
	> File Name: func.h
	> Author: honghao
	> Mail: honghao@shu.edu.cn 
	> Created Time: 2020年04月2日 星期四 14时25分17秒
 ************************************************************************/
#include "head.h"
//进行3维的 低秩  张量分解
int main(){
	ofstream ofs;

	ofs.open("result.txt", ios::out);

	for(int i=100; i<1300;i=i+100){
		warmupcu();
		// for(int i = 0; i < d1*d2*d3 ;++i)
		// 	cout << h_tensor[i] << "  " ;
		// cout<<endl;
		double duration = 0.0;
		// ttdec_gpu_float(h_tensor, h_n_array, h_r_array, eps);
		float time = 0.0;
		int calculateTimes = 10;
		// 	
		dt *h_tensor;
		int q = i, r = 50; 
		int d1 = q, d2 = q, d3 = q;
		// int d1 = 20, d2 = 20, d3 = 20;
		cout<<d1<<endl;

		// const int k = 8*7; //56
	 //    const int s = 8;
	 //    const int p = 2;
	 //    const int ks = k+s;
	    // float time = 0;
		int r_mul8 = 56;  //Multiple of 8
		if(i > 500){
			r_mul8 = q/10/8*8;
		}
		printf("r_mul8: %d\n", r_mul8);
		// cout << "r_mul8: "<< r_mul8 << endl;
		cudaHostAlloc((void**)&h_tensor,sizeof(dt)*d1*d2*d3,0);

		genTRTensor(h_tensor,d1,d2,d3,r);
		int h_n_array[3] = {d1, d2, d3};
		int h_r_array[4] = {2, r_mul8/2, 100, 2};
		double eps[2] = {1.e-4,1.e-8};
		float singleError[1]={0.0f}, singleComRatio[1] = {0.0f};

		trdec_gpu_float_unopt(h_tensor, h_n_array, h_r_array, eps);
		for(int i = 0; i < calculateTimes; ++i){
			cout << "---------------"<< i+1 << "-------------------" << endl;
			
			time = trdec_gpu_float_unopt(h_tensor, h_n_array, h_r_array, eps);
			
			duration += time;
		}
		duration = duration/(float)calculateTimes;
	    printf( "\n******************************\nTime: %f s\n******************************\n",  duration); 
		ofs << duration <<endl;
		cudaFreeHost(h_tensor);

		if(h_tensor) cudaFreeHost(h_tensor); h_tensor = NULL;
	}
	ofs.close();
	return 0;
}