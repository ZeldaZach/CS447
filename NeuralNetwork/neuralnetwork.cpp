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
const int testing_samples = 10'000, training_samples = 60'000, image_width = 28, image_height = 28;

// Neural network constraints
const int input_nodes = image_width * image_height, hidden_nodes = 128, output_nodes = 10;
const int epochs = 512;

// const floats, accessible from both host/device this way
#define learning_rate 0.001
#define momentum 0.9
#define epsilon 0.001

// Data containers
std::vector<std::vector<int>> images_2d;
int *images, *labels;

// Input layer -> Hidden layer
float w1[input_nodes * hidden_nodes];
__device__ float delta1[input_nodes * hidden_nodes], out1[input_nodes];

// Hidden layer -> Output layer
float w2[hidden_nodes * output_nodes];
__device__ float delta2[hidden_nodes * output_nodes], ihidden_nodes[hidden_nodes];
__device__ float out2[hidden_nodes], theta2[hidden_nodes];

// Output layer
__device__ float ioutput_nodes[output_nodes], out3[output_nodes], theta3[output_nodes], expected[output_nodes];

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
            // atomicAdd(&ihidden_nodes[j], out1[i] * w1[i * hidden_nodes + j]);
            ihidden_nodes[j] += out1[i] * w1[i * hidden_nodes + j];
        }
    }

    for (int i = 0; i < hidden_nodes; ++i) {
        out2[i] = activation_function(ihidden_nodes[i]);
    }

    for (int i = 0; i < hidden_nodes; ++i) {
        for (int j = 0; j < output_nodes; ++j) {
            // printf("ioutput_nodes[%d]=%f, adding %f*%f\n", j, ioutput_nodes[j], out2[i], w2[i * output_nodes + j]);
            // atomicAdd(&ioutput_nodes[j], out2[i] * w2[i * output_nodes + j]);
            ioutput_nodes[j] += out2[i] * w2[i * output_nodes + j];
        }
    }

    for (int i = 0; i < output_nodes; ++i) {
        out3[i] = activation_function(ioutput_nodes[i]);
        printf("out3[%d] = %f, ioutput_nodes[%d]=%f\n", i, out3[i], i, ioutput_nodes[i]);
    }
}

// Normalize error
__device__ float square_error()
{
    float res = 0.0;
    for (int i = 0; i < output_nodes; ++i) {
        res += (out3[i] - expected[i]) * (out3[i] - expected[i]);
    }
    res *= 0.5;
    return res;
}

__device__ void back_propagation(float *w1, float *w2)
{
    float sum;

    for (int i = 0; i < output_nodes; ++i) {
        theta3[i] = out3[i] * (1 - out3[i]) * (expected[i] - out3[i]);
    }

    for (int i = 0; i < hidden_nodes; ++i) {
        sum = 0.0;
        for (int j = 0; j < output_nodes; ++j) {
            sum += w2[i * output_nodes + j] * theta3[j];
        }
        theta2[i] = out2[i] * (1 - out2[i]) * sum;
    }

    for (int i = 0; i < hidden_nodes; ++i) {
        for (int j = 0; j < output_nodes; ++j) {
            delta2[i * output_nodes + j] =
                (learning_rate * theta3[j] * out2[i]) + (momentum * delta2[i * output_nodes + j]);

            atomicAdd(&w2[i * output_nodes + j], delta2[i * output_nodes + j]);
            // w2[i * output_nodes + j] += delta2[i * output_nodes + j];
        }
    }

    for (int i = 0; i < input_nodes; ++i) {
        for (int j = 0; j < hidden_nodes; j++) {
            delta1[i * hidden_nodes + j] =
                (learning_rate * theta2[j] * out1[i]) + (momentum * delta1[i * hidden_nodes + j]);

            atomicAdd(&w1[i * hidden_nodes + j], delta1[i * hidden_nodes + j]);
            // w1[i * hidden_nodes + j] += delta1[i * hidden_nodes + j];
        }
    }
}

__device__ int learning_process(float *w1, float *w2)
{
    // Initialize delta arrays
    for (int i = 0; i < input_nodes; ++i) {
        for (int j = 0; j < hidden_nodes; ++j) {
            delta1[i * hidden_nodes + j] = 0.0;
        }
    }

    for (int i = 0; i < hidden_nodes; ++i) {
        for (int j = 0; j < output_nodes; ++j) {
            delta2[i * output_nodes + j] = 0.0;
        }
    }

    for (int i = 0; i < epochs; ++i) {
        forward_learning(w1, w2);
        back_propagation(w1, w2);
        if (square_error() < epsilon) {
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

__global__ void training(float *w1, float *w2, int *images, int *labels)
{
    int sample = blockIdx.x * blockDim.x + threadIdx.x;

    if (sample % 100 == 0) {
        printf("Sample %d\n", sample);
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
    int nIterations = learning_process(w1, w2);

    if (sample % 100 == 0) {
        // Write down the squared error
        printf("Iterations: %d\n", nIterations);
        printf("Error: %0.6lf\n\n", square_error());
    }
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

    // for (int sample = 0; sample < testing_samples; ++sample) {

    // Getting (image, label)
    for (int j = 0; j < image_height; ++j) {
        for (int i = 0; i < image_width; ++i) {
            const int pos = j * image_width + i;
            out1[pos] = images[sample * input_nodes + pos];
        }
    }

    int label = labels[sample]; // read_image();

    for (int i = 0; i < output_nodes; ++i) {
        expected[i] = 0.0;
    }
    expected[label] = 1.0;

    // Classification - forward_learning procedure
    forward_learning(w1, w2);

    // Prediction
    int predict = 0;
    for (int i = 1; i < output_nodes; ++i) {
        printf("out3[%d]=%f, out3[%d]=%f\n", i, out3[i], predict, out3[predict]);
        if (out3[i] > out3[predict]) {
            predict = i;
        }
    }

    // Write down the classification result and the squared error
    float error = square_error();
    printf("Error: %0.6lf\n", error);

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

__global__ void kernel(float *w1)
{
    for (int i = 0; i < 10; i++) {
        printf("DEVICE w1[%d]=%d\n", i, w1[i]);
    }
}

int main(int argc, char **argv)
{
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
    init_nn_matrices();

    if (will_train) {
        image.open("mnist/train-images-idx3-ubyte", std::ios::in | std::ios::binary); // Binary image file
        label.open("mnist/train-labels-idx1-ubyte", std::ios::in | std::ios::binary); // Binary label file

        if (!image.is_open()) {
            std::cout << "MNIST training images not loaded" << std::endl;
            exit(1);
        }
        if (!label.is_open()) {
            std::cout << "MNIST labels not loaded" << std::endl;
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
        float *cuda_w1, *cuda_w2;
        int *cuda_images, *cuda_labels;

        cudaMalloc(&cuda_w1, input_nodes * hidden_nodes * sizeof(float));
        cudaMalloc(&cuda_w2, hidden_nodes * output_nodes * sizeof(float));
        cudaMalloc(&cuda_images, images_2d.size() * images_2d[0].size() * sizeof(int));
        cudaMalloc(&cuda_labels, images_2d.size() * sizeof(int));

        cudaMemcpy(cuda_w1, w1, hidden_nodes * output_nodes * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_w2, w2, input_nodes * hidden_nodes * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_images, images, images_2d.size() * images_2d[0].size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_labels, labels, images_2d.size() * sizeof(int), cudaMemcpyHostToDevice);

        training<<<60, 1000>>>(cuda_w1, cuda_w2, cuda_images, cuda_labels);
        cudaDeviceSynchronize();

        cudaMemcpy(w1, cuda_w1, hidden_nodes * output_nodes * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(w2, cuda_w2, input_nodes * hidden_nodes * sizeof(float), cudaMemcpyDeviceToHost);

        save_weights_to_file(model_fn);

        cudaFree(cuda_w1);
        cudaFree(cuda_w2);
        cudaFree(cuda_images);
        cudaFree(cuda_labels);

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
        cudaMalloc(&cuda_w2, hidden_nodes * output_nodes * sizeof(float));
        cudaMalloc(&cuda_images, images_2d.size() * images_2d[0].size() * sizeof(int));
        cudaMalloc(&cuda_labels, images_2d.size() * sizeof(int));
        cudaMalloc(&cuda_num_correct, sizeof(int));

        gpu_assert(cudaMemcpy(cuda_w1, w1, input_nodes * hidden_nodes * sizeof(float), cudaMemcpyHostToDevice));
        cudaMemcpy(cuda_w2, w2, hidden_nodes * output_nodes * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_images, images, images_2d.size() * images_2d[0].size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_labels, labels, images_2d.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_num_correct, new int(0), sizeof(int), cudaMemcpyHostToDevice);

        testing<<<10, 1000>>>(cuda_w1, cuda_w2, cuda_images, cuda_labels, cuda_num_correct);
        cudaDeviceSynchronize();

        int *result = new int(3);
        cudaMemcpy(result, cuda_num_correct, sizeof(int), cudaMemcpyDeviceToHost);

        float accuracy = (float)(*result) / testing_samples * 100.0;
        printf("Number of correct samples: %d/%d\n", *result, testing_samples);
        printf("Accuracy: %0.2lf\n", accuracy);

        cudaFree(cuda_w1);
        cudaFree(cuda_w2);
        cudaFree(cuda_images);
        cudaFree(cuda_labels);

        delete[] images;
        delete[] labels;
    }

    report.close();

    return 0;
}
