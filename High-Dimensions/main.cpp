#include <iostream>
#include <random>
#include <vector>
#include <iomanip>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

std::vector<int> sample_sphere(const int dimension) {
    assert(dimension > 0);

    std::default_random_engine generator{}; //std::chrono::system_clock::now().time_since_epoch().count());
    std::normal_distribution<> distribution(0.0, 1.0);

    std::vector<int> histogram;
    histogram.resize(100);

    // Run the experiment N times
    for (int i = 0; i < 1000000; i++) {
        // Determine our Gaussian variables
        std::vector<double> gaussian_values;
        gaussian_values.reserve(dimension);

        for (auto j = 0; j < dimension; j++) {
            gaussian_values.push_back(distribution(generator));
        }

        // https://karthikkaranth.me/blog/generating-random-points-in-a-sphere/
        // Normalize the values generated
        double sum = 0.0;
        for (const auto &t : gaussian_values) {
            sum += t * t;
        }
        double normalizer = std::sqrt(sum);
        double c = std::cbrt(distribution(generator));

        for (const double &t : gaussian_values) {
            double normalized_value = std::abs(t / normalizer * c);
            int index = static_cast<int>(normalized_value * 100);
            ++histogram[index];
        }
    }

    return histogram;

}

int main() {
    std::vector<std::vector<int>> dimensions;
    for (int dimension = 2; dimension <= 5; dimension++) {
        dimensions.push_back(sample_sphere(dimension));
        std::cout << "Finished Dimension " << dimension << std::endl;
    }

    for (int dimension = 2; dimension <= 5; dimension++) {
        plt::named_plot(std::to_string(dimension), dimensions.at(dimension - 2));
    }


    plt::title("Dimensions");
    plt::xlabel("Distance from sphere surface [0.00, 1.00]");
    plt::ylabel("Number of points at specific distance");
    plt::legend();
    plt::show();
}