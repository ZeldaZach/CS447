FLAGS = -Wall -Wextra -pedantic -g -pthread -std=c++14
HEADERS = KDTree.h KNearestNeighbors.h AtomicWriter.h
SOURCES = KDTree.cpp KNearestNeighbors.cpp main.cpp
EXEC = k-nn
DATA =
QUERY =
ARGS = /Users/zachary/Desktop/training/$(DATA).dat /Users/zachary/Desktop/training/$(QUERY).dat /Users/zachary/Desktop/training/results
VAL_ARGS = --leak-check=full

all: tree

tree: $(HEADERS) $(SOURCES)
	g++ $(SOURCES) -o $(EXEC) $(FLAGS)

run: tree
	./$(EXEC) $(ARGS)

test1: tree
	valgrind $(VAL_ARGS) ./$(EXEC) 1 $(ARGS)

test4: tree
	valgrind $(VAL_ARGS) ./$(EXEC) 4 $(ARGS)

clean:
	rm -f $(EXEC)
