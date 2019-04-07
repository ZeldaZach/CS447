//
// Created by Zachary Halpern on 2019-04-05.
//

#include <cassert>
#include <chrono>
#include <functional>
#include <immintrin.h>
#include <iostream>
#include <random>

// 16 million points
const int N = 16 * 1'000'000;

double time(const std::function<void()> &f)
{
    f(); // Run once to warm up.

    // Now time it for real.
    auto start = std::chrono::system_clock::now();
    f();
    auto stop = std::chrono::system_clock::now();

    return std::chrono::duration<double>(stop - start).count();
}

int main()
{
    // Point 1 storage
    alignas(32) static float axis_1_start[N], axis_2_start[N], axis_3_start[N], axis_4_start[N];

    // Point 2 storage
    alignas(32) static float axis_1_end[N], axis_2_end[N], axis_3_end[N], axis_4_end[N];

    std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<> distribution(-1.0, 1.0);

    // Generate the random points
    for (int i = 0; i < N; i++) {
        axis_1_start[i] = distribution(generator);
        axis_2_start[i] = distribution(generator);
        axis_3_start[i] = distribution(generator);
        axis_4_start[i] = distribution(generator);

        axis_1_end[i] = distribution(generator);
        axis_2_end[i] = distribution(generator);
        axis_3_end[i] = distribution(generator);
        axis_4_end[i] = distribution(generator);
    }

    // Sequential
    static float s_results[N];
    auto sequence = [&]() {
        for (int i = 0; i < N; i++) {
            // Euclidean distances
            const auto x1 = axis_1_start[i] - axis_1_end[i];
            const auto x2 = axis_2_start[i] - axis_2_end[i];
            const auto x3 = axis_3_start[i] - axis_3_end[i];
            const auto x4 = axis_4_start[i] - axis_4_end[i];

            // Euclidean distance calculation
            s_results[i] = std::sqrt((x1 * x1) + (x2 * x2) + (x3 * x3) + (x4 * x4));
        }
    };
    std::cout << "Sequential: " << (N / time(sequence)) / 1000000 << " Mops/s" << std::endl;

    // Parallel
    alignas(32) static float p_results[N];
    auto vec = [&]() {
        for (int i = 0; i < N / 8; i++) {
            // Start vectors
            __m256 ymm_axis_1_start = _mm256_load_ps(axis_1_start + 8 * i);
            __m256 ymm_axis_2_start = _mm256_load_ps(axis_2_start + 8 * i);
            __m256 ymm_axis_3_start = _mm256_load_ps(axis_3_start + 8 * i);
            __m256 ymm_axis_4_start = _mm256_load_ps(axis_4_start + 8 * i);

            // End vectors
            __m256 ymm_axis_1_end = _mm256_load_ps(axis_1_end + 8 * i);
            __m256 ymm_axis_2_end = _mm256_load_ps(axis_2_end + 8 * i);
            __m256 ymm_axis_3_end = _mm256_load_ps(axis_3_end + 8 * i);
            __m256 ymm_axis_4_end = _mm256_load_ps(axis_4_end + 8 * i);

            // Distances for Euclidean formula
            const auto x1_dist = ymm_axis_1_start - ymm_axis_1_end;
            const auto x2_dist = ymm_axis_2_start - ymm_axis_2_end;
            const auto x3_dist = ymm_axis_3_start - ymm_axis_3_end;
            const auto x4_dist = ymm_axis_4_start - ymm_axis_4_end;

            // Euclidean distance on vectors
            const auto x1_x_x1 = _mm256_mul_ps(x1_dist, x1_dist);
            const auto x2_x_x2 = _mm256_mul_ps(x2_dist, x2_dist);
            const auto x3_x_x3 = _mm256_mul_ps(x3_dist, x3_dist);
            const auto x4_x_x4 = _mm256_mul_ps(x4_dist, x4_dist);

            // Euclidean results
            __m256 ymm_results = _mm256_sqrt_ps(x1_x_x1 + x2_x_x2 + x3_x_x3 + x4_x_x4);

            // Store back to the vector
            _mm256_store_ps(p_results + 8 * i, ymm_results);
        }
    };
    std::cout << "Vector: " << (N / time(vec)) / 1000000 << " Mops/s" << std::endl;

    // Validation check
    for (int i = 0; i < N; i++) {
        // Small differences can happen with extreme optimizations
        if (std::abs(s_results[i] - p_results[i]) > 0.001) {
            assert(false);
        }
    }
}