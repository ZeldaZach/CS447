//
// Created by Zachary Halpern on 2019-03-04.
//

#include "KNearestNeighbors.h"
#include <iomanip>
#include <iostream>
#include <string>

/**
 * Main method -- Start K-NN Parallel
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv)
{
    if (argc != 5) {
        std::cerr << "Usage: ./k-nn <n_cores> <training_file> <query_file> <result_file>" << std::endl;
        exit(1);
    }

    std::cout << "Threads: " << std::strtoul(argv[1], nullptr, 10) << std::endl;
    KNearestNeighbors knn(std::strtoul(argv[1], nullptr, 10), argv[2], argv[3], argv[4]);
}
