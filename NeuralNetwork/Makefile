all: format cpp cu

cpp: format neuralnetwork.cpp
	g++ neuralnetwork.cpp -o cpu_training -std=c++17 -Wall -Wextra -pedantic -O2

cu: format neuralnetwork.cu
	nvcc neuralnetwork.cu -o gpu_training -arch=compute_30 -I"/mnt/c/ProgramData/NVIDIA Corporation/CUDA Samples/v10.1/common/inc/"

format:
	clang-format -i -style=file *.cpp

clean:
	rm -f cpu_training gpu_training