PY_LIB = -I/usr/include/python2.7 -I/usr/local/lib/python2.7/site-packages/numpy/core/include -DWITHOUT_NUMPY -lpython2.7
FLAGS = -Wall -Wextra -pedantic -std=c++14 -O

# Compiler chosen based on computer type
UNAME := $(shell uname)
ifeq ($(UNAME), Linux)
CC = g++
endif
ifeq ($(UNAME), Darwin)
CC = g++-8
endif


all: make_sphere make_segments

make_sphere: sphere.cpp matplotlibcpp.h
	$(CC) sphere.cpp $(FLAGS) $(PY_LIB) -fopenmp -o assgn_2_1

make_segments: segments.cpp
	$(CC) segments.cpp $(FLAGS) -mavx2 -o assgn_2_2

clean:
	rm -f assgn_2_1 assgn_2_2

.PHONY: make_sphere make_segments clean all
