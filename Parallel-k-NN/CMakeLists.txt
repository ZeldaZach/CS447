cmake_minimum_required(VERSION 3.13)
project(Parallel_k_NN)

set(CMAKE_CXX_STANDARD 14)

add_executable(Parallel_k_NN main.cpp KNearestNeighbors.cpp KNearestNeighbors.h KDTree.cpp KDTree.h)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")