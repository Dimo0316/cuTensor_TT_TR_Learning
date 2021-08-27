cc=/usr/local/cuda-10.1/bin/nvcc
des=full
 
#source = ht_init.cu  htd_tensorcore.cu kernel.cu 
#source = ht_init.cu htd.cu kernel.cu  
source = cudaver.cu init.cpp 

link = -lcublas -lcusolver -lcurand -lcufft -std=c++14

$(des):$(source)
	$(cc) -o $(des) $(source) $(link)

clean:
	rm -rf $(des)