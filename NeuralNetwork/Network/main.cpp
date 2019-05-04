#include <cassert>
#include <cstdlib>
#include <ctime>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <unistd.h>

#include "Network.h"

using namespace std;

float stepFunction(float x)
{
    if (x > 0.9) {
        return 1.0;
    }
    if (x < 0.1) {
        return 0.0;
    }
    return x;
}

void swap(int &i)
{
    // Some of the & are superfluous.
    i = (0xff & (i >> 24)) | (0xff00 & (i >> 8)) | (0xff0000 & (i << 8)) | (0xff000000 & (i << 24));
}

int read_int(int fd)
{
    int rv;
    int i;
    rv = read(fd, &i, 4);
    assert(rv == 4);
    swap(i);
    return i;
}

void read_mnist_labels(const char *fn, std::vector<std::vector<float>> &output, int N)
{
    output.resize(N);

    int fd = open(fn, O_RDONLY);
    assert(fd >= 0);

    int magic = read_int(fd);
    assert(magic == 0x801);

    int n_labels = read_int(fd);
    assert(n_labels == N);

    unsigned char(labels)[60'000];
    int rv = read(fd, labels, N);
    assert(rv == N);
    for (int i = 0; i < N; i++) {
        output[i].resize(10);
        output[i][(unsigned int)labels[i]] = 1;
    }

    rv = close(fd);
    assert(rv == 0);
}

void read_mnist_images(const char *fn, std::vector<std::vector<float>> &input, int N)
{
    input.resize(N);

    int fd = open(fn, O_RDONLY);
    assert(fd >= 0);

    int magic = read_int(fd);
    assert(magic == 0x803);

    int n_images = read_int(fd);
    assert(n_images == N);

    int n_rows = read_int(fd);
    assert(n_rows == 28);

    int n_cols = read_int(fd);
    assert(n_cols == 28);

    int rv;
    for (int i = 0; i < N; i++) {
        unsigned char tmp[28][28];
        rv = read(fd, tmp, 28 * 28);
        assert(rv == 28 * 28);
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                // Make go from -1 to 1.
                // input[i][r][c] = float(tmp[r][c])/127.5 - 1;
                input[i].push_back(float(tmp[r][c]) / 127.5 - 1);
            }
        }
    }

    rv = close(fd);
    assert(rv == 0);
}

int main()
{
    std::vector<std::vector<float>> trainInput, trainOutput, testInput, testOutput;
    read_mnist_images("../ken/mnist/train-images-idx3-ubyte", trainInput, 60'000);
    read_mnist_labels("../ken/mnist/train-labels-idx1-ubyte", trainOutput, 60'000);
    read_mnist_images("../ken/mnist/t10k-images-idx3-ubyte", testInput, 10'000);
    read_mnist_labels("../ken/mnist/t10k-labels-idx1-ubyte", testOutput, 10'000);

    //std::ofstream output_file("./mine_training.txt");
    //std::ostream_iterator<unsigned char> output_iterator(output_file, "");
    //std::copy(trainInput.begin(), trainInput.end(), output_iterator);

    // 28*28 = input neurons (images are 28x28 pixels)
    // 1024 hidden neurons
    // 10 output neurons (0-9)
    // 0.6 learning rate (0.4 dropoff)
   /*
    Network net({28 * 28, 1024, 10}, 0.6);

    // train on 30 iterations
    cout << "Training..." << endl;
    for (unsigned int i = 0; i < 3; i++) {
        for (unsigned int j = 0; j < 60'000; j++)
        {
            net.computeOutput(trainInput[j]);
            net.learn(trainOutput[j]);
            if (j % 100 == 0) {
                std::cout << "#" << j+1 << "/" << "60000" << std::endl;
            }
        }
        cout << "#" << i + 1 << "/5" << endl;
    }*/

    Network net("params");

    // test
    cout << endl << "Testing..." << endl;
    cout << "expected output : actual output" << endl << endl;
    for (unsigned int i = 0; i < 100; /*testInput.size();*/ i++) // testing on last 10 examples
    {
        for (unsigned int j = 0; j < 10; j++) {
            cout << testOutput[i][j] << " ";
        }
        cout << ": " << net.computeOutput(testInput[i]).applyFunction(stepFunction) << endl;
        // as the sigmoid function never reaches 0.0 nor 1.0
        // it can be a good idea to consider values greater than 0.9 as 1.0 and values smaller than 0.1 as 0.0
        // hence the step function.
    }

    cout << endl << "Saving parameters...";
    net.saveNetworkParams("params");
    cout << "ok!" << endl;

    // net.loadNetworkParams("params"); or Network net("params");
}
