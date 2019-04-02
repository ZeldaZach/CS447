#include <iostream>
#include <random>
#include <vector>
#include <iomanip>

void sample_sphere(const int dimension) {
    assert(dimension > 0);

    std::default_random_engine generator{};//std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<> distribution(0.0, 1.0);

    std::vector<int> histogram;
    histogram.resize(100);

    // Run the experiment N times
    for (int i = 0; i < 1000; i++) {
        // Determine our Gaussian variables
        std::vector<double> gaussian_values;
        gaussian_values.reserve(dimension);

        for (auto j = 0; j < dimension; j++) {
            gaussian_values.push_back(distribution(generator));
        }

        // Normalize the values generated
        double sum = 0.0;
        for (const auto &t : gaussian_values) {
            sum += t * t;
        }
        double normalizer = std::sqrt(sum);

        for (const double &t : gaussian_values) {
            double normalized_value = std::abs(t / normalizer);
            int index = static_cast<int>(normalized_value * 100);
            histogram.at(index)++;
        }
    }

    for (int i = 0; i < histogram.size(); i++) {
        std::cout << std::setw(4) << i * 0.01 << ":";
        for (int j = 0; j < histogram.at(i); j++) {
            std::cout << "*";
        }
        std::cout << std::endl;
    }
}

int main() {
    sample_sphere(16);
    return 0;
}