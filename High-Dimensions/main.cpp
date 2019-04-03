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
 * @return Histogram vector
 */
std::vector<int> sample_sphere(const int dimension)
{
    assert(dimension > 0);

    std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
    std::normal_distribution<> distribution(0.0, 1.0);

    std::vector<int> histogram;
    histogram.resize(100);

    auto begin = std::chrono::steady_clock::now();

#pragma omp parallel for
    // Run the experiment N times
    for (int i = 0; i < 10000; i++) {
        double sum;

        do {
            // Determine our Gaussian variables
            std::vector<double> gaussian_values;
            gaussian_values.reserve(dimension);

            sum = 0.0;
            for (auto j = 0; j < dimension; j++) {
                const double random_num = distribution(generator);

                gaussian_values.push_back(random_num);
                sum += random_num * random_num;
            }
        } while (sum > 1.0);

        ++histogram[static_cast<int>(std::sqrt(sum) * 100)];
    }

    auto end = std::chrono::steady_clock::now();
    std::cout << "Time to build dimension " << dimension << ": "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;

    return histogram;
}

/**
 * Main method to start the program
 * @param argc # Args
 * @param argv Args
 */
int main(int argc, char **argv)
{
    // Set max threads if necessary
    if (argc > 2) {
        omp_set_num_threads(std::atoi(argv[1]));
    }

    // Calculate and plot the histogram
    for (int dimension = 2; dimension <= 11; dimension++) {
        const auto t = sample_sphere(dimension);
        plt::named_plot(std::to_string(dimension), t);
    }

    // More plot stuff
    plt::title("Hypersphere Point Check");
    plt::xlabel("Distance from Sphere Center * 100");
    plt::ylabel("Points at Specific Distance");
    plt::legend();
    plt::show();
}