#include <cmath>
#include <cstdio>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define gpu_assert(rv) gpu_assert_h((rv), __FILE__, __LINE__)
void gpu_assert_h(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

// Output files
const char *model_fn = "neural_network_matrices.txt";
const char *report_fn = "training_report.txt";

// Data constraints
const int testing_samples = 500, training_samples = 500, image_width = 28, image_height = 28;

// Neural network constraints
const int input_nodes = image_width * image_height, hidden_nodes = 128, output_nodes = 10;
const int epochs = 512;

__device__ float cuda_w1[input_nodes * hidden_nodes], cuda_w2[hidden_nodes * output_nodes], out2[hidden_nodes];

// const floats, accessible from both host/device this way
#define learning_rate 0.001
#define momentum 0.9
#define epsilon 0.001

// Data containers
std::vector<std::vector<int>> images_2d;
int *images, *labels;

// Input layer -> Hidden layer
// float w1[input_nodes * hidden_nodes];

// Hidden layer -> Output layer
// float w2[hidden_nodes * output_nodes];

float *w1, *w2;

// File stream to read/write data
std::ifstream image, label;
std::ofstream report;

template <typename T> T *vectorToArray(std::vector<std::vector<T>> const &v)
{
    T *rv = (T *)malloc((v.size() * v[0].size()) * sizeof(T)); // Assuming all rows have the same size
    for (unsigned i = 0; i < v.size(); i++)
        memcpy(rv + v[i].size() * i, &(v[i][0]), v[i].size() * sizeof(T));
    return rv;
}

void project_details()
{
    std::cout << "Input neurons: " << input_nodes << std::endl;
    std::cout << "Hidden neurons: " << hidden_nodes << std::endl;
    std::cout << "Output neurons: " << output_nodes << std::endl;
    std::cout << std::endl;
    std::cout << "Epochs: " << epochs << std::endl;
    std::cout << "Learning rate: " << learning_rate << std::endl;
}

void init_nn_matrices()
{
    // Initialization of weights from Input layer to Hidden layer
    for (int i = 0; i < input_nodes; ++i) {
        for (int j = 0; j < hidden_nodes; ++j) {
            int sign = rand() % 2;
            w1[i * hidden_nodes + j] = (float)(rand() % 10 + 1) / (10 * hidden_nodes);
            if (sign == 1) {
                w1[i * hidden_nodes + j] = -w1[i * hidden_nodes + j];
            }
        }
    }

    // Initialization of weights from Hidden layer to Output layer
    for (int i = 0; i < hidden_nodes; ++i) {
        for (int j = 0; j < output_nodes; ++j) {
            int sign = rand() % 2;
            w2[i * output_nodes + j] = (float)(rand() % 10 + 1) / (10.0 * output_nodes);
            if (sign == 1) {
                w2[i * output_nodes + j] = -w2[i * output_nodes + j];
            }
        }
    }
}

void load_model_from_backup(std::string file_name)
{
    std::ifstream file(file_name, std::ios::in);

    if (!file.is_open()) {
        printf("Unable to open matrix file\n");
        exit(1);
    }

    // read_image layer -> Hidden layer
    for (int i = 0; i < input_nodes; ++i) {
        for (int j = 0; j < hidden_nodes; ++j) {
            file >> w1[i * hidden_nodes + j];

            if (j % 100 == 0) {
                // printf("Loaded value w1[%d]=%f\n", i * hidden_nodes + j, w1[i * hidden_nodes + j]);
            }
        }
    }

    // Hidden layer -> Output layer
    for (int i = 0; i < hidden_nodes; ++i) {
        for (int j = 0; j < output_nodes; ++j) {
            file >> w2[i * output_nodes + j];
            // printf("Loaded value w2[%d]=%f\n", i * output_nodes + j, w2[i * output_nodes + j]);
        }
    }

    file.close();
}

__device__ float activation_function(float x)
{
    // SIGMOID
    // printf("Activation %f to %f\n", x, 1.0/(1.0+exp(-x)));
    return 1.0 / (1.0 + exp(-x));
}

__device__ void forward_learning(float *w1,
                                 float *w2,
                                 float *ihidden_nodes1,
                                 float *ioutput_nodes1,
                                 float *out31,
                                 float *out11,
                                 bool use_local = false)
{
    int sample = blockIdx.x * blockDim.x + threadIdx.x;

    float tmp_out2[hidden_nodes];

    for (int i = 0; i < hidden_nodes; ++i) {
        ihidden_nodes1[i] = 0.0;
    }

    for (int i = 0; i < output_nodes; ++i) {
        ioutput_nodes1[i] = 0.0;
        out31[i] = 0.0;
    }

    for (int i = 0; i < input_nodes; ++i) {
        for (int j = 0; j < hidden_nodes; ++j) {
            ihidden_nodes1[j] += out11[i] * w1[i * hidden_nodes + j];
            printf("[%d] ihidden_nodes1[%d] += %f * %f\n", sample, j, out11[i], w1[i * hidden_nodes + j]);
        }
    }

    for (int i = 0; i < hidden_nodes; ++i) {
        if (!use_local) {
            out2[i] = activation_function(ihidden_nodes1[i]);
        } else {
            tmp_out2[i] = activation_function(ihidden_nodes1[i]);
        }
    }

    for (int i = 0; i < hidden_nodes; ++i) {
        for (int j = 0; j < output_nodes; ++j) {
            // printf("ioutput_nodes1[%d]=%f, adding %f*%f\n", j, ioutput_nodes1[j], out2[i], w2[i * output_nodes + j]);

            if (!use_local) {
                ioutput_nodes1[j] += out2[i] * w2[i * output_nodes + j];
            } else {
                ioutput_nodes1[j] += tmp_out2[i] * w2[i * output_nodes + j];
                printf("[%d] ioutput_nodes1[%d] += %f * %f\n", sample, j, tmp_out2[i], w2[i * output_nodes + j]);
            }
        }
    }

    for (int i = 0; i < output_nodes; ++i) {
        out31[i] = activation_function(ioutput_nodes1[i]);
        printf("[%d] out31[%d] = %f, ioutput_nodes1[%d]=%f\n", sample, i, out31[i], i, ioutput_nodes1[i]);
    }
}

/*
__device__ void forward_learning(float *w1, float *w2)
{
    for (int i = 0; i < hidden_nodes; ++i) {
        ihidden_nodes[i] = 0.0;
    }

    for (int i = 0; i < output_nodes; ++i) {
        ioutput_nodes[i] = 0.0;
        out3[i] = 0.0;
    }

    for (int i = 0; i < input_nodes; ++i) {
        for (int j = 0; j < hidden_nodes; ++j) {
            ihidden_nodes[j] += out1[i] * w1[i * hidden_nodes + j];
        }
    }

    for (int i = 0; i < hidden_nodes; ++i) {
        out2[i] = activation_function(ihidden_nodes[i]);
    }

    for (int i = 0; i < hidden_nodes; ++i) {
        for (int j = 0; j < output_nodes; ++j) {
            // printf("ioutput_nodes[%d]=%f, adding %f*%f\n", j, ioutput_nodes[j], out2[i], w2[i * output_nodes + j]);
            ioutput_nodes[j] += out2[i] * w2[i * output_nodes + j];
        }
    }

    for (int i = 0; i < output_nodes; ++i) {
        out3[i] = activation_function(ioutput_nodes[i]);
        // printf("out3[%d] = %f, ioutput_nodes[%d]=%f\n", i, out3[i], i, ioutput_nodes[i]);
    }
}
*/

// Normalize error
__device__ float square_error(float *out3, float *expected)
{
    float res = 0.0;
    for (int i = 0; i < output_nodes; ++i) {
        res += (out3[i] - expected[i]) * (out3[i] - expected[i]);
    }
    res *= 0.5;
    return res;
}

__device__ void back_propagation(float *out3, float *out1, float *expected, float *delta2, float *delta1)
{
    float sum;

    float theta2[hidden_nodes];
    float theta3[output_nodes];

    for (int i = 0; i < output_nodes; ++i) {
        theta3[i] = out3[i] * (1 - out3[i]) * (expected[i] - out3[i]);
    }

    for (int i = 0; i < hidden_nodes; ++i) {
        sum = 0.1;
        for (int j = 0; j < output_nodes; ++j) {
            sum += cuda_w2[i * output_nodes + j] * theta3[j];
        }
        theta2[i] = out2[i] * (1 - out2[i]) * sum;
    }

    int sample = blockIdx.x * blockDim.x + threadIdx.x;

    printf("[%d] W2 before=%f\n", sample, cuda_w2[0]);

    for (int i = 0; i < hidden_nodes; ++i) {
        for (int j = 0; j < output_nodes; ++j) {
            delta2[i * output_nodes + j] =
                (learning_rate * theta3[j] * out2[i]) + (momentum * delta2[i * output_nodes + j]);

            if (i == 0 && j == 0) {
                printf("[%d] W2 ADDING (%f * %f * %f) + (%f * %f)\n", sample, learning_rate, theta3[j], out2[i],
                       momentum, delta2[i * output_nodes + j]);
            }

            atomicAdd(&cuda_w2[i * output_nodes + j], delta2[i * output_nodes + j]);

            // w2[i * output_nodes + j] += delta2[i * output_nodes + j];
        }
    }
    printf("[%d] W2 after=%f\n", sample, cuda_w2[0]);

    for (int i = 0; i < input_nodes; ++i) {
        for (int j = 0; j < hidden_nodes; j++) {
            delta1[i * hidden_nodes + j] =
                (learning_rate * theta2[j] * out1[i]) + (momentum * delta1[i * hidden_nodes + j]);

            atomicAdd(&cuda_w1[i * hidden_nodes + j], delta1[i * hidden_nodes + j]);
            // w1[i * hidden_nodes + j] += delta1[i * hidden_nodes + j];
        }
    }
}

__device__ int learning_process(float *ihidden_nodes, float *ioutput_nodes, float *out3, float *out1, float *expected)
{
    float delta2[hidden_nodes * output_nodes];
    float delta1[input_nodes * hidden_nodes];

    for (int i = 0; i < epochs; ++i) {
        forward_learning(cuda_w1, cuda_w2, ihidden_nodes, ioutput_nodes, out3, out1);
        back_propagation(out3, out1, expected, delta2, delta1);
        if (square_error(out3, expected) < epsilon) {
            return i;
        }
    }
    return epochs;
}

void read_images(int sample_count)
{
    char number;

    delete labels;
    labels = new int[sample_count];
    images_2d.clear();
    images_2d.resize(sample_count);

    for (int image_number = 0; image_number < sample_count; image_number++) {
        // Create image holders, set values to 0 by default
        std::vector<int> tmp_img;
        tmp_img.resize(image_width * image_height);

        for (int height = 0; height < image_height; height++) {
            for (int width = 0; width < image_width; width++) {
                image.read(&number, sizeof(char));
                if (number == 0) {
                    tmp_img.at(height * image_width + width) = 0;
                } else {
                    tmp_img.at(height * image_width + width) = 1;
                }
            }
        }
        images_2d.at(image_number) = tmp_img;

        // Read in label
        label.read(&number, sizeof(char));
        labels[image_number] = number;
    }
}

void save_weights_to_file(std::string file_name)
{
    std::ofstream file(file_name.c_str(), std::ios::out);

    // read_image layer -> Hidden layer
    for (int i = 0; i < input_nodes; ++i) {
        for (int j = 0; j < hidden_nodes; ++j) {
            file << w1[i * hidden_nodes + j] << " ";
        }
        file << std::endl;
    }

    // Hidden layer -> Output layer
    for (int i = 0; i < hidden_nodes; ++i) {
        for (int j = 0; j < output_nodes; ++j) {
            file << w2[i * output_nodes + j] << " ";
        }
        file << std::endl;
    }

    file.close();
}

__global__ void kernel()
{
    printf("Hello Kernel %d\n", blockIdx.x * blockDim.x + threadIdx.x);
}

__global__ void training(int *images, int *labels)
{
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= training_samples) {
        return;
    }

    float out1[input_nodes];
    float ihidden_nodes[hidden_nodes];
    float ioutput_nodes[output_nodes], out3[output_nodes], expected[output_nodes];

    for (int i = 0; i < input_nodes; i++) {
        out1[i] = 0.0;
    }
    for (int i = 0; i < hidden_nodes; i++) {
        ihidden_nodes[i] = 0.0;
        out2[i] = 0.0;
    }
    for (int i = 0; i < output_nodes; i++) {
        ioutput_nodes[i] = 0.0;
        out3[i] = 0.0;
        expected[i] = 0.0;
    }

    // Getting (image, label)
    for (int j = 0; j < image_height; ++j) {
        for (int i = 0; i < image_width; ++i) {
            const int pos = j * image_width + i;
            out1[pos] = images[sample * input_nodes + pos];
        }
    }

    for (int i = 0; i < output_nodes; ++i) {
        expected[i] = 0.0;
    }
    expected[labels[sample]] = 1.0;

    // Learning process: forward_learning (Forward procedure) - Back propagation
    int nIterations = learning_process(ihidden_nodes, ioutput_nodes, out3, out1, expected);

    /*
    for (int i = 0; i < input_nodes * hidden_nodes; i++) {
        w1[i] = cuda_w1[i];
    }
    for (int i = 0; i < hidden_nodes * output_nodes; i++) {
        w2[i] = cuda_w2[i];
    }
     */

    /*
    if (sample % 100 == 0) {
        // Write down the squared error
        printf("Iterations: %d\n", nIterations);
        printf("Error: %0.6lf\n\n", square_error(out3, expected));
    }
     */
    // report << "Sample " << sample + 1 << ": Iterations = " << nIterations << ", Error = " << square_error()
    //       << std::endl;

    // Save the current network (weights) every so often
    // if (sample % 100 == 0) {
    //    save_weights_to_file(model_fn);
    //}

    // Save the final network
    // save_weights_to_file(model_fn);
}

__global__ void testing(float *w1, float *w2, int *images, int *labels, int *nCorrect)
{
    int sample = blockIdx.x * blockDim.x + threadIdx.x;

    if (sample >= testing_samples) {
        return;
    }

    /*
    if (sample == 118) {
        for (int i = 0; i < image_width * image_height; i++) {
            for (int height = 1; height <= image_height; height++) {
                for (int width = 1; width <= image_width; width++) {
                    printf("%d", images[i * image_height * image_width + height * image_width + width]);
                }
                printf("\n");
            }

            printf("%d\n\n\n", labels[i]);
        }
    }*/

    float ihidden_nodes1[hidden_nodes];
    float ioutput_nodes1[output_nodes];
    float out31[output_nodes];
    float out11[input_nodes];
    // float expected[output_nodes];

    // for (int sample = 0; sample < testing_samples; ++sample) {

    // Getting (image, label)
    for (int j = 0; j < image_height; ++j) {
        for (int i = 0; i < image_width; ++i) {
            const int pos = j * image_width + i;
            out11[pos] = images[sample * image_height*image_width + pos];

            if (sample == 118) {
                printf("[118] out11[%d]=%f\n", pos, out11[pos]);
            }
        }
    }

    int label = labels[sample]; // read_image();

    // Classification - forward_learning procedure
    forward_learning(w1, w2, ihidden_nodes1, ioutput_nodes1, out31, out11, true);
    // forward_learning(w1, w2);

    // Prediction
    int predict = 0;
    for (int i = 1; i < output_nodes; ++i) {
        printf("[%d] out3[%d]=%f, out3[%d]=%f\n", sample, i, out31[i], predict, out31[predict]);
        if (out31[i] > out31[predict]) {
            predict = i;
        }
    }

    // Write down the classification result and the squared error
    // float error = square_error();
    // printf("Error: %0.6lf\n", error);

    if (label == predict) {
        atomicAdd(nCorrect, 1);
        printf("Classification: YES. Label = %d. Prediction = %d\n", label, predict);
        // printf("Sample %d: YES. Label = %d. Prediction = %d\n", sample, label, predict);
    } else {
        printf("Classification: NO. Label = %d. Prediction = %d\n", label, predict);
        // printf("Sample %d: NO. Label = %d. Prediction = %d\n", sample, label, predict);
    }
    //}

    // Summary
    // float accuracy = (float)(*nCorrect) / testing_samples * 100.0;
    // printf("Number of correct samples: %d/%d\n", *nCorrect, testing_samples);
    // printf("Accuracy: %0.2lf\n", accuracy);

    // report << "Number of correct samples: " << nCorrect << " / " << testing_samples << std::endl;
    // report << "Accuracy: " << accuracy << std::endl;
}

int main(int argc, char **argv)
{
    gpu_assert(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024000 * 10 * 10));

    bool will_test = false, will_train = false;
    if (argc == 1) {
        will_train = true;
        will_test = true;
    } else if (argc == 2) {
        // strcmp (key,buffer) != 0
        if (strcmp(argv[1], "train") == 0) {
            will_train = true;
        } else if (strcmp(argv[1], "test") == 0) {
            will_test = true;
        } else if (strcmp(argv[1], "both") == 0) {
            will_train = true;
            will_test = true;
        } else {
            std::cout << "Usage: " << argv[0] << " ['train', 'test', 'both']" << std::endl;
            exit(1);
        }
    }

    project_details();

    report.open(report_fn, std::ios::out);

    // Neural Network Initialization
    cudaMallocManaged(&w1, input_nodes * hidden_nodes * sizeof(float));
    cudaMallocManaged(&w2, hidden_nodes * output_nodes * sizeof(float));
    init_nn_matrices();

    if (will_train) {
        image.open("mnist/train-images-idx3-ubyte", std::ios::in | std::ios::binary); // Binary image file
        label.open("mnist/train-labels-idx1-ubyte", std::ios::in | std::ios::binary); // Binary label file

        if (!image.is_open()) {
            printf("MNIST training images not loaded");
            exit(1);
        }
        if (!label.is_open()) {
            printf("MNIST labels not loaded");
            exit(1);
        }

        // Reading file headers
        char number;
        for (int i = 1; i <= 16; ++i) {
            image.read(&number, sizeof(char));
        }
        for (int i = 1; i <= 8; ++i) {
            label.read(&number, sizeof(char));
        }

        // Read images all at once, into memory (cuda memory)
        // Then we can access vai index in the read_image method
        // From there,

        // Read all images & labels into memory
        read_images(training_samples);
        image.close();
        label.close();

        images = vectorToArray<int>(images_2d);

        // CUDA MALLOCS
        int *cuda_images, *cuda_labels;

        gpu_assert(cudaMalloc(&cuda_images, images_2d.size() * images_2d[0].size() * sizeof(int)));
        gpu_assert(cudaMalloc(&cuda_labels, images_2d.size() * sizeof(int)));

        gpu_assert(cudaMemcpy(cuda_images, images, images_2d.size() * images_2d[0].size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        gpu_assert(cudaMemcpy(cuda_labels, labels, images_2d.size() * sizeof(int), cudaMemcpyHostToDevice));
        gpu_assert(
            cudaMemcpyToSymbol(cuda_w1, w1, input_nodes * hidden_nodes * sizeof(float), 0, cudaMemcpyHostToDevice));
        gpu_assert(
            cudaMemcpyToSymbol(cuda_w2, w2, hidden_nodes * output_nodes * sizeof(float), 0, cudaMemcpyHostToDevice));

        // kernel<<<10, 10>>>();
        training<<<50, 10>>>(cuda_images, cuda_labels);
        cudaDeviceSynchronize();

        for (int i = 0; i < 10; i++) {
            printf("w2[%d]=%f\n", i, w2[i]);
        }

        gpu_assert(
            cudaMemcpyFromSymbol(w1, cuda_w1, input_nodes * hidden_nodes * sizeof(float), 0, cudaMemcpyDeviceToHost));
        gpu_assert(
            cudaMemcpyFromSymbol(w2, cuda_w2, hidden_nodes * output_nodes * sizeof(float), 0, cudaMemcpyDeviceToHost));

        for (int i = 0; i < 10; i++) {
            printf("w2[%d]=%f\n", i, w2[i]);
        }

        save_weights_to_file(model_fn);

        gpu_assert(cudaFree(cuda_images));
        gpu_assert(cudaFree(cuda_labels));

        delete[] images;
        delete[] labels;
    }

    if (will_test) {
        image.open("mnist/t10k-images-idx3-ubyte", std::ios::in | std::ios::binary); // Binary image file
        label.open("mnist/t10k-labels-idx1-ubyte", std::ios::in | std::ios::binary); // Binary label file

        // Reading file headers
        char number;
        for (int i = 1; i <= 16; ++i) {
            image.read(&number, sizeof(char));
        }
        for (int i = 1; i <= 8; ++i) {
            label.read(&number, sizeof(char));
        }

        read_images(testing_samples);
        image.close();
        label.close();

        images = vectorToArray<int>(images_2d);

        // Load model (weight matrices) of a trained Neural Network
        load_model_from_backup(model_fn);

        // CUDA MALLOCS
        float *cuda_w1, *cuda_w2;
        int *cuda_images, *cuda_labels, *cuda_num_correct;

        gpu_assert(cudaMalloc(&cuda_w1, input_nodes * hidden_nodes * sizeof(float)));
        gpu_assert(cudaMalloc(&cuda_w2, hidden_nodes * output_nodes * sizeof(float)));
        gpu_assert(cudaMalloc(&cuda_images, images_2d.size() * images_2d[0].size() * sizeof(int)));
        gpu_assert(cudaMalloc(&cuda_labels, images_2d.size() * sizeof(int)));
        gpu_assert(cudaMalloc(&cuda_num_correct, sizeof(int)));

        gpu_assert(cudaMemcpy(cuda_w1, w1, input_nodes * hidden_nodes * sizeof(float), cudaMemcpyHostToDevice));
        gpu_assert(cudaMemcpy(cuda_w2, w2, hidden_nodes * output_nodes * sizeof(float), cudaMemcpyHostToDevice));
        gpu_assert(cudaMemcpy(cuda_images, images, images_2d.size() * images_2d[0].size() * sizeof(int),
                              cudaMemcpyHostToDevice));
        gpu_assert(cudaMemcpy(cuda_labels, labels, images_2d.size() * sizeof(int), cudaMemcpyHostToDevice));
        gpu_assert(cudaMemcpy(cuda_num_correct, new int(0), sizeof(int), cudaMemcpyHostToDevice));

        testing<<<5, 100>>>(cuda_w1, cuda_w2, cuda_images, cuda_labels, cuda_num_correct);
        cudaDeviceSynchronize();

        int *result = new int(3);
        gpu_assert(cudaMemcpy(result, cuda_num_correct, sizeof(int), cudaMemcpyDeviceToHost));

        float accuracy = (float)(*result) / testing_samples * 100.0;
        printf("Number of correct samples: %d/%d\n", *result, testing_samples);
        printf("Accuracy: %0.2lf\n", accuracy);

        gpu_assert(cudaFree(cuda_w1));
        gpu_assert(cudaFree(cuda_w2));
        gpu_assert(cudaFree(cuda_images));
        gpu_assert(cudaFree(cuda_labels));

        delete[] images;
        delete[] labels;
    }

    report.close();

    return 0;
}
