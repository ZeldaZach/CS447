#include "matplotlibcpp.h"
#include <chrono>
#include <cstdlib>
#include <fstream>
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
 * @param is_parallel to execute in OpenMP or not
 * @return Histogram vector and time to execute
 */
std::pair<std::vector<int>, unsigned long>
sample_sphere(const int dimension, const int sample_points, const bool is_parallel)
{
    assert(dimension > 0);

    static thread_local unsigned int seed = time(nullptr);

    // static thread_local std::mt19937 generator(std::chrono::system_clock::now().time_since_epoch().count());
    // std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
    // std::normal_distribution<> distribution(0.0, 1.0);

    std::vector<int> histogram(100);

    const auto begin = std::chrono::steady_clock::now();

// http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
#pragma omp parallel for num_threads(omp_get_max_threads()) shared(histogram) if (is_parallel)
    for (int i = 0; i < sample_points; i++) {
        double normalizer = 0.0, sum = 0.0;
        std::vector<double> gaussian_values(dimension + 2);

        for (int d = 0; d < dimension + 2; d++) {
            // gaussian_values.push_back(distribution(generator));
            gaussian_values.push_back(rand_r(&seed) / (double)RAND_MAX);
            normalizer += gaussian_values.back() * gaussian_values.back();
        }
        gaussian_values.pop_back();
        gaussian_values.pop_back();

        normalizer = std::sqrt(normalizer);
        for (auto &value : gaussian_values) {
            value /= normalizer;
            sum += value * value;
        }

        sum = std::sqrt(sum);
        ++histogram[static_cast<int>(std::sqrt(sum) * 100)];
    }
    const auto end = std::chrono::steady_clock::now();

    // Print out timing results
    if (is_parallel) {
        std::cout << " (Parallel x" << omp_get_max_threads() << ") ";
    } else {
        std::cout << "(Not Parallel) ";
    }

    const unsigned long time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    std::cout << "Dimension " << dimension << " with " << sample_points << " nodes: " << time << "ms" << std::endl;
    return std::make_pair<>(histogram, time);
}

/**
 * Main method to start the program
 * @param argc # Args
 * @param argv Args
 */
int main(int argc, char **argv)
{
    if (argc == 1) {
        std::cout << "Usage: ./hypersphere <max_omp_threads> [sphere_points] ['true' out to file] ['true' for graph]"
                  << std::endl;
        exit(1);
    }

    unsigned int node_points = 1000;
    bool show_gui_graph = false;

    // Redirect std::cout to file for future usages
    std::ofstream out("hypersphere_output.txt");
    std::streambuf *coutbuf = std::cout.rdbuf();

    // Set max threads if necessary
    if (argc >= 2) {
        omp_set_num_threads(std::strtol(argv[1], nullptr, 10));
        if (argc >= 3) {
            node_points = std::strtol(argv[2], nullptr, 10);
            if (argc >= 4) {
                if (strcmp(argv[3], "true") == 0) {
                    std::cout.rdbuf(out.rdbuf());
                }
                if (argc >= 5) {
                    show_gui_graph = (strcmp(argv[4], "true") == 0);
                }
            }
        }
    } else {
        omp_set_num_threads(1);
    }

    std::cout << "Using " << omp_get_max_threads() << " threads" << std::endl;

    // Holder to show the histogram
    std::vector<std::vector<int>> histograms;
    histograms.reserve(15);

    unsigned long run_time = 0;
    // Calculate histogram for non-parallel for timing
    for (int dimension = 2; dimension <= 16; dimension++) {
        const auto histogram_and_time = sample_sphere(dimension, node_points, false);
        run_time += histogram_and_time.second;
    }

    // Calculate and plot the histogram (for parallel version)
    unsigned long p_run_time = 0;
    for (int dimension = 2; dimension <= 16; dimension++) {
        const auto histogram_and_time = sample_sphere(dimension, node_points, true);
        histograms.push_back(histogram_and_time.first);
        p_run_time += histogram_and_time.second;
    }

    // Print timetables
    std::cout << "Non-Parallel computation time: " << run_time << "ms" << std::endl;
    std::cout << "    Parallel computation time: " << p_run_time << "ms" << std::endl;

    // Create a histogram plot setup
    for (unsigned long i = 0; i < histograms.size(); i++) {
        plt::named_plot(std::to_string(i + 2), histograms.at(i));

        std::cout << "\nDimension " << i + 2 << " Distance from center histogram" << std::endl;
        for (unsigned long j = 0; j < histograms.at(i).size(); j++) {
            std::cout << std::setw(4) << 0.01 * j << ": " << 1.0 * histograms.at(i).at(j) / node_points * 100 << "%"
                      << std::endl;
        }
    }

    std::cout.rdbuf(coutbuf); // reset to standard output again
    if (show_gui_graph) {
        plt::title("Hypersphere Point Check");
        plt::xlabel("Distance from Sphere Center * 100");
        plt::ylabel("Points at Specific Distance");
        plt::legend();
        plt::show();
    }
}