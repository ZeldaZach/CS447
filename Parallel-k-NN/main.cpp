//
// Created by Zachary Halpern on 2019-03-04.
//

#include "KNearestNeighbors.h"
#include <iomanip>
#include <iostream>
#include <string>

void printVector(const std::vector<float> &v)
{
    for (float i : v) {
        std::cout << std::setw(10) << i;
    }
    std::cout << std::endl;
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        std::cerr << "Usage: ./k-nn <n_cores> <training_file> <query_file> <result_file>" << std::endl;
        exit(1);
    }

    KNearestNeighbors knn;

    // Training data
    knn.readFile(argv[1]);

    // Query data
    knn.readFile(argv[2]);

    /*
    std::cout << "POINTS IN FILE" << std::endl;
    for (const auto &t : knn.getPoints()) {
        printVector(t);
    }

    for (const auto &test_point : knn.getQueries()) {
        auto result = knn.getNearestNeighbor(test_point);

        std::cout << "Test Point:";
        printVector(test_point);

        std::cout << "Closest Point:";
        printVector(result);
    }
    */

    // Generate and output results
    std::string output_file = knn.generateAndWriteResults(argv[3]);
    knn.readFile(const_cast<char *>(output_file.c_str()));
}
