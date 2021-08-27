cc=/usr/local/cuda/bin/nvcc
des=tt_full

source = main.cpp cudaver.cu getImg.cpp layers.cu net.cu 

link = -lcublas -lcusolver -lcurand -lcufft -std=c++14

$(des):$(source)
	$(cc) -o $(des) $(source) $(link)

clean:
	rm -rf $(des)