#cc=/usr/local/cuda-10.1/bin/nvcc
cc = nvcc
des=dmrg
source = dmrg.cu doDMRG.cu fuc.cu lanczos.cu
 
link = -lcublas -lcusolver -lcurand -lcutensor -std=c++11

$(des):$(source)
	$(cc) -o $(des) $(source) $(link)

clean:
	rm -rf $(des)
