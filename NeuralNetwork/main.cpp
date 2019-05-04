//
// Created by ZaHal on 5/2/2019.
//

#include "main.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <unistd.h>

template <int N> void NeuralNetwork::read_mnist_images(const std::string &fn, float (&imgs)[N][28][28])
{
    int rv;

    int fd;
    fd = open(fn.c_str(), O_RDONLY);
    assert(fd >= 0);

    int magic = read_int(fd);
    assert(magic == 0x803);

    int n_images = read_int(fd);
    assert(n_images == N);

    int n_rows = read_int(fd);
    assert(n_rows == 28);

    int n_cols = read_int(fd);
    assert(n_cols == 28);

    for (int i = 0; i < N; i++) {
        unsigned char tmp[28][28];
        rv = read(fd, tmp, 28 * 28);
        assert(rv == 28 * 28);
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                // Make go from -1 to 1.
                imgs[i][r][c] = double(tmp[r][c]) / 127.5 - 1;
            }
        }
    }

    rv = close(fd);
    assert(rv == 0);
}

template <int N> void NeuralNetwork::read_mnist_labels(const std::string &fn, unsigned char (&labels)[N])
{
    int rv;

    int fd = open(fn.c_str(), O_RDONLY);
    assert(fd >= 0);

    int magic = read_int(fd);
    assert(magic == 0x801);

    int n_labels = read_int(fd);
    assert(n_labels == N);

    rv = read(fd, labels, N);
    assert(rv == N);
    for (int i = 0; i < N; i++) {
        assert(labels[i] >= 0 && labels[i] <= 9);
    }

    rv = close(fd);
    assert(rv == 0);
}

int NeuralNetwork::read_int(int fd)
{
    int rv;
    int i;
    rv = read(fd, &i, 4);
    assert(rv == 4);
    swap(i);
    return i;
}

void NeuralNetwork::swap(int &i)
{
    // Some of the & are superfluous.
    i = (0xff & (i >> 24)) | (0xff00 & (i >> 8)) | (0xff0000 & (i << 8)) | (0xff000000 & (i << 24));
}

#if 0
void NeuralNetwork::read_input_data(float training_images[][28][28],
                                    unsigned char training_labels[],
                                    float test_images[][28][28],
                                    unsigned char test_labels[])
{
    read_mnist_images("mnist/train-images-idx3-ubyte", training_images);
    output_pgm("img0.pgm", training_images[0]);
    output_pgm("img59999.pgm", training_images[59999]);

    read_mnist_labels("mnist/train-labels-idx1-ubyte", training_labels);
    assert(training_labels[0] == 5);
    assert(training_labels[59'999] == 8);

    read_mnist_images("mnist/t10k-images-idx3-ubyte", test_images);
    read_mnist_labels("mnist/t10k-labels-idx1-ubyte", test_labels);
}
#endif

int main()
{
    static float training_images[60'000][28][28];
    static unsigned char training_labels[60'000];
    static float test_images[10'000][28][28];
    static unsigned char test_labels[10'000];

    auto nn = new NeuralNetwork();

    // Read MNIST training data in to system
    nn->read_mnist_images("mnist/train-images-idx3-ubyte", training_images);
    nn->read_mnist_labels("mnist/train-labels-idx1-ubyte", training_labels);
    assert(training_labels[0] == 5);
    assert(training_labels[59'999] == 8);
    nn->read_mnist_images("mnist/t10k-images-idx3-ubyte", test_images);
    nn->read_mnist_labels("mnist/t10k-labels-idx1-ubyte", test_labels);

    std::default_random_engine eng(9815);
    std::uniform_int_distribution<size_t> pick_test(0, 9'999);

    for (int epoch = 0; epoch < 30; epoch++) {
        // Create shuffled sequence of training images.
        std::vector<int> training(60'000);
        std::iota(training.begin(), training.end(), 0);
        assert(*--training.end() == 59'999);
        std::shuffle(training.begin(), training.end(), eng);

        float *training_images_cuda, *training_labels_cuda;
        cudaMalloc(&training_images_cuda, sizeof(training_images) * 28 * 28 * sizeof(float));
        cudaMalloc(&training_labels_cuda, sizeof(training_labels) * sizeof(unsigned char));

        cudaMemcpy(training_images_cuda, training_images, sizeof(training_images) * 28 * 28 * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(training_labels_cuda, training_labels, sizeof(training_labels) * sizeof(unsigned char),
                   cudaMemcpyHostToDevice);

        // train(training_images_cuda, training_labels_cuda);
        /*
        for (int r = 0; r < 600; r++) {
            for (int s = 0; s < 100; s ++) {
                //train(training_images, training_labels);
            }
        }*/

        cudaMemcpy(training_images, training_images_cuda, sizeof(training_images) * 28 * 28 * sizeof(float),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(training_labels, training_labels_cuda, sizeof(training_labels) * sizeof(unsigned char),
                   cudaMemcpyDeviceToHost);

        cudaFree(training_images_cuda);
        cudaFree(training_labels_cuda);
    }

    delete nn;
    return 0;
}
