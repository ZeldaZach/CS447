all: cpp

cpp: training_nn.cpp
	g++ training_nn.cpp -o cpu_training -std=c++17 -Wall -Wextra -pedantic -g

cu: training_nn.cpp
	nvcc training_nn.cpp -o gpu_training
	
clean:
	rm -f cpu_training gpu_training