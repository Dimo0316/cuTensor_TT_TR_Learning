cc=/usr/bin/nvcc
des=trdec_unopt2

source =  main.cpp  kernel.cu trdec_gpu_float_unopt2_rank.cu
#source =  main.cpp  kernel.cu ttdec_gpu_double.cu 
link = -lcublas -lcusolver -lcurand -std=c++11

$(des):$(source)
	$(cc) $(link) $(source) -o $(des) 

clean:
	rm -rf $(des)
