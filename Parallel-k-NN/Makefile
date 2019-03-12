FLAGS = -Wall -Wextra -pedantic -pthread -std=c++14 -O -g
FLAGS_STATIC = -Wall -Wextra -pedantic -std=c++14 -O -g -pg -static -pthread -Wl,--whole-archive -lpthread -Wl,--no-whole-archive
HEADERS = KDTree.h KNearestNeighbors.h AtomicWriter.h
SOURCES = KDTree.cpp KNearestNeighbors.cpp main.cpp
EXEC = k-nn
DATA = data_12153728
QUERY = query_12154023
ARGS = training/$(DATA).dat training/$(QUERY).dat training/results
VAL_ARGS = --leak-check=full

all: tree

tree_static: $(HEADERS) $(SOURCES)
	g++ $(SOURCES) -o $(EXEC) $(FLAGS_STATIC)

tree: $(HEADERS) $(SOURCES)
	g++ $(SOURCES) -o $(EXEC) $(FLAGS)

run1: tree
	./$(EXEC) 1 $(ARGS)

run4: tree
	./$(EXEC) 4 $(ARGS)

test1: tree
	valgrind $(VAL_ARGS) ./$(EXEC) 1 $(ARGS)

test4: tree
	valgrind $(VAL_ARGS) ./$(EXEC) 4 $(ARGS)

time1: tree
	time ./$(EXEC) 1 $(ARGS)

time4: tree
	time ./$(EXEC) 4 $(ARGS)

clean:
	rm -f $(EXEC)
