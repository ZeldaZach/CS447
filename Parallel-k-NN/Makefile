FLAGS = -Wall -Wextra -pedantic -g -std=c++14 -stdlib=libc++
HEADERS = KDTree.h KNearestNeighbors.h
SOURCES = KDTree.cpp KNearestNeighbors.cpp main.cpp
EXEC = k-nn
ARGS = ./training/data_7161234.dat ./training/query_7161251.dat ./training/output
VAL_ARGS = --leak-check=full

all: tree

tree: $(HEADERS) $(SOURCES)
	g++ $(SOURCES) -o $(EXEC) $(FLAGS)

run: tree
	./$(EXEC) $(ARGS)

test: tree
	valgrind $(VAL_ARGS) ./$(EXEC) $(ARGS)

clean:
	rm -f $(EXEC)
