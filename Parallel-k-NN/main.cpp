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
    std::cout << "POINTS IN FILE" << std::endl;
    for (const auto &t : knn.points) {
        for (const auto u : t) {
            std::cout << std::setw(15) << u;
        }
        std::cout << std::endl;
    }

    /*knn.readFile(std::string(argv[2]));


    std::cout << std::endl;

    for (const auto &t : knn.queries) {
        for (const auto u : t) {
            std::cout << std::setw(15) << u;
        }
        std::cout << std::endl;
    }*/

    std::cout << "\nCLOSEST POINT TO 0.0" << std::endl;
    // Create tree and test
    knn.create_tree();
    std::vector<float> test_float({100.0, 27.0});
    auto result = knn.getNearestNeighbor(test_float);
    for (float &i : result) {
        std::cout << std::setw(15) << i;
    }
    std::cout << std::endl;

    // knn.writeResults(argv[3]);
}