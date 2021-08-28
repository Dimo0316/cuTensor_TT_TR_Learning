/*************************************************************************
	> File Name: main.cpp
	> Author: honghao
	> Mail: honghao@shu.edu.cn 
	> Created Time: 2020年04月2日 星期四 14时25分17秒
 ************************************************************************/
#include "head.h"


//进行3维的 稠密  张量分解
int main(){
	ofstream ofs;

	// ofs.open("result.txt", ios::out);

	for(int i=1000; i<2000;i=i+100){
		dt *h_tensor;
		srand(time(NULL));
		int k = i, r = 50; 
		int d1 = k, d2 = k, d3 = k;

		cout<<d1<<endl;
		double eps = 1.e-7;
		int r_mul8 = 56;  //Multiple of 8
		if(i > 500){
			r_mul8 = i/10/8*8;
		}

		cudaHostAlloc((void**)&h_tensor,sizeof(dt)*d1*d2*d3,0);
	
		genTTTensor(h_tensor,d1,d2,d3,r);
		// printMatrix(d1, d2*d3, h_tensor, d1, "h_tensor");
		int h_n_array[3] = {d1, d2, d3};
		int h_r_array[4] = {1, r_mul8, r_mul8, 1};

		warmupcu();
		cudaDeviceSynchronize();
		ttdec_gpu_float_unopt(h_tensor, h_n_array, h_r_array, eps);
	    
		float duration = 0.0;

		float time = 0.0;
		int calculateTimes = 10;
		for(int i = 0; i < calculateTimes; ++i){
			cout << "---------------"<< i+1 << "-------------------" << endl;
			time = ttdec_gpu_float_unopt(h_tensor, h_n_array, h_r_array, eps);
			duration += time;
		}

		//获取时间
		duration = duration/(float)calculateTimes;
		printf( "\n******************************\nTime: %f s\n******************************\n", duration); 
	    // printf( "\ntime: %f s\n", duration ); 
		// ofs << duration <<endl;
		cudaFreeHost(h_tensor);
	}
	// ofs.close();
	return 0;
}