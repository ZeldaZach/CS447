PY_LIB = -I/usr/include/python2.7 -I/usr/local/lib/python2.7/site-packages/numpy/core/include/ -DWITHOUT_NUMPY -lpython2.7
FLAGS = -Wall -Wextra -pedantic -std=c++14 -O
CC = g++-8

all: hsphere lsegments

hsphere: sphere.cpp matplotlibcpp.h
	$(CC) sphere.cpp $(FLAGS) $(PY_LIB) -fopenmp -o hypersphere

lsegments: segments.cpp
	$(CC) segments.cpp $(FLAGS) -mavx2 -o linesegments

clean:
	rm -f hypersphere linesegments

.PHONY: hsphere lsegments clean