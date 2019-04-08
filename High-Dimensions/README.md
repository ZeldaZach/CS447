# Assignment-2
CS 447/547 2019 S: Assignment 2
http://www.cs.binghamton.edu/~kchiu/cs547/prog/2/

## Compile
```bash
make
```

## Usage
```bash
./assgn_2_1 <max_omp_threads> <sphere_points> <dimensions> <"file"> <"graph">
./assgn_2_2
```

## Best tests
```bash
./assgn_2_1 8 1000000 16 file none    # standard solve
./assgn_2_1 8 1000000 16 file graph   # +10 bonus points
./assgn_2_1 8 1000000 50 file graph   # +15 bonus points

./assgn_2_2  # standard solve
```

### Hypersphere References:
- [ResearchGate](https://www.researchgate.net/post/Is_there_a_way_to_specify_how_many_cores_a_program_should_run-in_other_words_can_I_control_where_the_threads_are_mapped)
- [Waterloo](http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf)
- [Wikipedia](https://en.wikipedia.org/wiki/Normal_distribution)
- [WolframAlpha](http://mathworld.wolfram.com/HyperspherePointPicking.html)

### Segments References:
- [Professor](http://www.cs.binghamton.edu/~kchiu/cs547/prog/2/)