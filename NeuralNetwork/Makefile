all: format cpp cu

cpp: format neuralnetwork_sequential.cpp
	g++ neuralnetwork_sequential.cpp -o cpu_neural -std=c++17 -Wall -Wextra -pedantic -O2

cu: format neuralnetwork_parallel.cu
	nvcc neuralnetwork_parallel.cu -o gpu_neural -arch=compute_30

format:
	clang-format -i -style=file *.cpp *.cu

clean:
	rm -f cpu_neural gpu_neural