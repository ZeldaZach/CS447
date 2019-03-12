FLAGS = -Wall -Wextra -pedantic -std=c++14 -O -g -pthread
FLAGS_STATIC = $(FLAGS) -pg -static -Wl,--whole-archive -lpthread -Wl,--no-whole-archive
HEADERS = KDTree.h KNearestNeighbors.h AtomicWriter.h
SOURCES = KDTree.cpp KNearestNeighbors.cpp main.cpp
VAL_ARGS = --leak-check=full
EXEC = k-nn

DATA = FileName.dat
QUERY = FileName.dat
ARGS = training/$(DATA) training/$(QUERY) training/results.dat

all: dynamic_tree

static_tree: $(HEADERS) $(SOURCES)
	g++ $(SOURCES) -o $(EXEC) $(FLAGS_STATIC)

dynamic_tree: $(HEADERS) $(SOURCES)
	g++ $(SOURCES) -o $(EXEC) $(FLAGS)

run1: dynamic_tree
	./$(EXEC) 1 $(ARGS)

run4: dynamic_tree
	./$(EXEC) 4 $(ARGS)

time1: dynamic_tree
	time ./$(EXEC) 1 $(ARGS)

time4: dynamic_tree
	time ./$(EXEC) 4 $(ARGS)

valgrind1: dynamic_tree
	valgrind $(VAL_ARGS) ./$(EXEC) 1 $(ARGS)

valgrind4: dynamic_tree
	valgrind $(VAL_ARGS) ./$(EXEC) 4 $(ARGS)

clean:
	rm -f $(EXEC)

