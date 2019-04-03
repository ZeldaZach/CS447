LIBS = -I/usr/include/python2.7 -I/usr/local/lib/python2.7/site-packages/numpy/core/include/
FLAGS = -O4 -g -Wall -Wextra -pedantic -DWITHOUT_NUMPY -lpython2.7 -std=c++14 $(LIBS)
CC = g++-8

parallel:
	$(CC) main.cpp $(FLAGS) -fopenmp -o hypersphere

single:
	$(CC) main.cpp $(FLAGS) -o hypersphere

clean:
	rm -f hypersphere