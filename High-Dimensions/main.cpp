#include "matplotlibcpp.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <random>
#include <vector>

namespace plt = matplotlibcpp;

/**
 * Get a histogram vector for points distance from center
 * of a sphere in K dimension
 * @param dimension Dimension to test
 * @param sample_points How many points to generate on sphere
 * @return Histogram vector
 */
std::vector<int> sample_sphere(const int dimension, const int sample_points)
{
    assert(dimension > 0);

    std::default_random_engine generator; //(std::chrono::system_clock::now().time_since_epoch().count());
    std::normal_distribution<> distribution(0.0, 1.0);

    std::vector<int> histogram(100);

    const auto begin = std::chrono::steady_clock::now();

#pragma omp parallel for shared(histogram)
    // Run the experiment N times
    for (int i = 0; i < sample_points; i++) {
        double sum;
        std::vector<double> gaussian_values(dimension);

        do {
            // Determine our Gaussian variables
            gaussian_values.clear();
            sum = 0;

            for (auto j = 0; j < dimension; j++) {
                const double random_num = distribution(generator);

                gaussian_values.push_back(random_num);
                sum += random_num * random_num;
            }
        } while (sum > 1);

        ++histogram[static_cast<int>(std::sqrt(sum) * 100)];
    }

    const auto end = std::chrono::steady_clock::now();
    std::cout << "Time to build dimension " << dimension << " with " << sample_points
              << " nodes: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms"
              << std::endl;

    return histogram;
}

/**
 * Main method to start the program
 * @param argc # Args
 * @param argv Args
 */
int main(int argc, char **argv)
{
    unsigned int node_points = 1000;

    // Set max threads if necessary
    if (argc >= 2) {
        omp_set_num_threads(std::strtol(argv[1], nullptr, 10));
        if (argc >= 3) {
            node_points = std::strtol(argv[2], nullptr, 10);
        }
    } else {
        omp_set_num_threads(1);
    }

    std::cout << "Using " << omp_get_max_threads() << " threads" << std::endl;

    // Calculate and plot the histogram
    for (int dimension = 2; dimension <= 9; dimension++) {
        const auto histogram = sample_sphere(dimension, node_points);
        plt::named_plot(std::to_string(dimension), histogram);
    }

    // More plot stuff
    plt::title("Hypersphere Point Check");
    plt::xlabel("Distance from Sphere Center * 100");
    plt::ylabel("Points at Specific Distance");
    plt::legend();
    plt::clf();
    // plt::show();
}