//
// Created by Zachary Halpern on 2019-03-04.
//

#include "KNearestNeighbors.h"
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char **argv)
{
    if (argc != 4) {
        std::cerr << "Usage: Invalid" << std::endl;
        exit(1);
    }

    KNearestNeighbors knn;

    knn.readFile(std::string(argv[1]));
    knn.readFile(std::string(argv[2]));

    for (const auto &t : knn.points) {
        for (const auto u : t) {
            std::cout << std::setw(15) << u;
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;

    for (const auto &t : knn.queries) {
        for (const auto u : t) {
            std::cout << std::setw(15) << u;
        }
        std::cout << std::endl;
    }

    knn.writeResults(argv[3]);
}