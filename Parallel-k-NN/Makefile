FLAGS = -Wall -Wextra -pedantic -g -std=c++14
HEADERS = KDTree.h KNearestNeighbors.h
SOURCES = KDTree.cpp KNearestNeighbors.cpp main.cpp
EXEC = k-nn

all: tree

tree: $(HEADERS) $(SOURCES)
	g++ $(SOURCES) -o $(EXEC) $(FLAGS)

run: tree
	./$(EXEC)

clean:
	rm -f $(EXEC)
