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

    // Create tree and test
    knn.create_tree();

    std::vector<std::vector<float>> test_points({{880.0, -350.0}, {0, 0}, {-1000, -500}});

    for (const auto &test_point : test_points) {
        auto result = knn.getNearestNeighbor(test_point);

        std::cout << "Test Point:";
        for (float i : test_point) {
            std::cout << std::setw(15) << i;
        }
        std::cout << std::endl;

        std::cout << "Closest Point:";
        for (float i : result) {
            std::cout << std::setw(15) << i;
        }
        std::cout << std::endl;
    }

    /*
    Test Point:            880           -350
    Closest Point:         884.06       -363.149
    Test Point:              0              0
    Closest Point:        100.213        27.4071
    Test Point:          -1000           -500
    Closest Point:       -911.686       -619.303
     */

    // knn.writeResults(argv[3]);
}