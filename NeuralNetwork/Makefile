all: cpp

cpp: neuralnetwork.cpp
	g++ neuralnetwork.cpp -o cpu_training -std=c++17 -Wall -Wextra -pedantic -g

cu: neuralnetwork.cpp
	nvcc neuralnetwork.cpp -o gpu_training

format:
	clang-format -i -style=file *.cpp

clean:
	rm -f cpu_training gpu_training