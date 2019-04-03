cmake_minimum_required(VERSION 3.13)
project(High_Dimensions)

set(CMAKE_CXX_STANDARD 14)
set(CPLUS_INCLUDE_PATH /System/Library/Frameworks/Python.framework/)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O -Wall -Wextra -pedantic -DWITHOUT_NUMPY -I/usr/include/python2.7 -I/usr/local/lib/python2.7/site-packages/numpy/core/include/ -lpython2.7")

add_executable(High_Dimensions main.cpp matplotlibcpp.h)