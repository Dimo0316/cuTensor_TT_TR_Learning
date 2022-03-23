cc=/usr/bin/nvcc
des=trdec2 

source =  main.cpp  kernel.cu trdec_gpu_float_rsvd_lu_half.cu trals_gpu_float.cu
# source =  main.cpp  kernel.cu trdec_gpu_float_2.cu trals_gpu_float.cu
#source =  main.cpp  kernel.cu ttdec_gpu_float.cu
#source =  main.cpp  kernel.cu ttdec_gpu_double.cu 
link = -lcublas -lcusolver -lcurand -std=c++11

$(des):$(source)
	$(cc) $(link) $(source) -o $(des) 

clean:
	rm -rf $(des)
