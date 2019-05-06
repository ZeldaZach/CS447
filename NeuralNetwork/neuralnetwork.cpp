#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <string>

const char *model_fn = "neural_network_matrices.txt";
const char *report_fn = "training-report.dat";

// Image constraints
const int testing_samples = 10000, training_samples = 60000, image_width = 28, image_height = 28;

// Neural network constraints
const int input_nodes = image_width * image_height, hidden_nodes = 128, output_nodes = 10;
const int epochs = 512;
const float learning_rate = 0.001;
const float momentum = 0.9;
const float epsilon = 1e-3;

// read_image layer -> Hidden layer
float *w1[input_nodes + 1], *delta1[input_nodes + 1], *out1;

// Hidden layer -> Output layer
float *w2[hidden_nodes + 1], *delta2[hidden_nodes + 1], *ihidden_nodes, *out2, *theta2;

// Output layer
float *ioutput_nodes, *out3, *theta3;
float expected[output_nodes + 1];

// Image. In MNIST: 28x28 gray scale images.
int img[image_width + 1][image_height + 1];

// File stream to read data (image, label) and write down a report
std::ifstream image;
std::ifstream label;
std::ofstream report;

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
    // Layer 1 - Layer 2 = read_image layer - Hidden layer
    for (int i = 1; i <= input_nodes; ++i) {
        w1[i] = new float[hidden_nodes + 1];
        delta1[i] = new float[hidden_nodes + 1];
    }

    out1 = new float[input_nodes + 1];

    // Layer 2 - Layer 3 = Hidden layer - Output layer
    for (int i = 1; i <= hidden_nodes; ++i) {
        w2[i] = new float[output_nodes + 1];
        delta2[i] = new float[output_nodes + 1];
    }

    ihidden_nodes = new float[hidden_nodes + 1];
    out2 = new float[hidden_nodes + 1];
    theta2 = new float[hidden_nodes + 1];

    // Layer 3 - Output layer
    ioutput_nodes = new float[output_nodes + 1];
    out3 = new float[output_nodes + 1];
    theta3 = new float[output_nodes + 1];

    // Initialization of weights from read_image layer to Hidden layer
    for (int i = 1; i <= input_nodes; ++i) {
        for (int j = 1; j <= hidden_nodes; ++j) {
            int sign = rand() % 2;
            w1[i][j] = (float)(rand() % 10 + 1) / (10 * hidden_nodes);
            if (sign == 1) {
                w1[i][j] = -w1[i][j];
            }
        }
    }

    // Initialization of weights from Hidden layer to Output layer
    for (int i = 1; i <= hidden_nodes; ++i) {
        for (int j = 1; j <= output_nodes; ++j) {
            int sign = rand() % 2;
            w2[i][j] = (float)(rand() % 10 + 1) / (10.0 * output_nodes);
            if (sign == 1) {
                w2[i][j] = -w2[i][j];
            }
        }
    }
}

void load_model_from_backup(std::string file_name)
{
    std::ifstream file(file_name, std::ios::in);

    // read_image layer -> Hidden layer
    for (int i = 1; i <= input_nodes; ++i) {
        for (int j = 1; j <= hidden_nodes; ++j) {
            file >> w1[i][j];
        }
    }

    // Hidden layer -> Output layer
    for (int i = 1; i <= hidden_nodes; ++i) {
        for (int j = 1; j <= output_nodes; ++j) {
            file >> w2[i][j];
        }
    }

    file.close();
}

float activation_function(float x)
{
    // SIGMOID
    return 1.0 / (1.0 + exp(-x));
}

void forward_learning()
{
    for (int i = 1; i <= hidden_nodes; ++i) {
        ihidden_nodes[i] = 0.0;
    }

    for (int i = 1; i <= output_nodes; ++i) {
        ioutput_nodes[i] = 0.0;
    }

    for (int i = 1; i <= input_nodes; ++i) {
        for (int j = 1; j <= hidden_nodes; ++j) {
            ihidden_nodes[j] += out1[i] * w1[i][j];
        }
    }

    for (int i = 1; i <= hidden_nodes; ++i) {
        out2[i] = activation_function(ihidden_nodes[i]);
    }

    for (int i = 1; i <= hidden_nodes; ++i) {
        for (int j = 1; j <= output_nodes; ++j) {
            ioutput_nodes[j] += out2[i] * w2[i][j];
        }
    }

    for (int i = 1; i <= output_nodes; ++i) {
        out3[i] = activation_function(ioutput_nodes[i]);
    }
}

// Normalize error
float square_error()
{
    float res = 0.0;
    for (int i = 1; i <= output_nodes; ++i) {
        res += (out3[i] - expected[i]) * (out3[i] - expected[i]);
    }
    res *= 0.5;
    return res;
}

void back_propagation()
{
    float sum;

    for (int i = 1; i <= output_nodes; ++i) {
        theta3[i] = out3[i] * (1 - out3[i]) * (expected[i] - out3[i]);
    }

    for (int i = 1; i <= hidden_nodes; ++i) {
        sum = 0.0;
        for (int j = 1; j <= output_nodes; ++j) {
            sum += w2[i][j] * theta3[j];
        }
        theta2[i] = out2[i] * (1 - out2[i]) * sum;
    }

    for (int i = 1; i <= hidden_nodes; ++i) {
        for (int j = 1; j <= output_nodes; ++j) {
            delta2[i][j] = (learning_rate * theta3[j] * out2[i]) + (momentum * delta2[i][j]);
            w2[i][j] += delta2[i][j];
        }
    }

    for (int i = 1; i <= input_nodes; ++i) {
        for (int j = 1; j <= hidden_nodes; j++) {
            delta1[i][j] = (learning_rate * theta2[j] * out1[i]) + (momentum * delta1[i][j]);
            w1[i][j] += delta1[i][j];
        }
    }
}

int learning_process()
{
    // Initialize delta arrays
    for (int i = 1; i <= input_nodes; ++i) {
        for (int j = 1; j <= hidden_nodes; ++j) {
            delta1[i][j] = 0.0;
        }
    }

    for (int i = 1; i <= hidden_nodes; ++i) {
        for (int j = 1; j <= output_nodes; ++j) {
            delta2[i][j] = 0.0;
        }
    }

    for (int i = 1; i <= epochs; ++i) {
        forward_learning();
        back_propagation();
        if (square_error() < epsilon) {
            return i;
        }
    }
    return epochs;
}

// Read image
int read_image()
{
    // Reading image
    char number;
    for (int j = 1; j <= image_height; ++j) {
        for (int i = 1; i <= image_width; ++i) {
            image.read(&number, sizeof(char));
            if (number == 0) {
                img[i][j] = 0;
            } else {
                img[i][j] = 1;
            }
        }
    }

    for (int j = 1; j <= image_height; ++j) {
        for (int i = 1; i <= image_width; ++i) {
            int pos = i + (j - 1) * image_width;
            out1[pos] = img[i][j];
        }
    }

    // Reading label
    label.read(&number, sizeof(char));
    for (int i = 1; i <= output_nodes; ++i) {
        expected[i] = 0.0;
    }
    expected[number + 1] = 1.0;

    std::cout << "Label: " << (int)(number) << std::endl;
    return (int)number;
}

void save_weights_to_file(std::string file_name)
{
    std::ofstream file(file_name.c_str(), std::ios::out);

    // read_image layer -> Hidden layer
    for (int i = 1; i <= input_nodes; ++i) {
        for (int j = 1; j <= hidden_nodes; ++j) {
            file << w1[i][j] << " ";
        }
        file << std::endl;
    }

    // Hidden layer -> Output layer
    for (int i = 1; i <= hidden_nodes; ++i) {
        for (int j = 1; j <= output_nodes; ++j) {
            file << w2[i][j] << " ";
        }
        file << std::endl;
    }

    file.close();
}

void training()
{
    for (int sample = 1; sample <= training_samples; ++sample) {
        std::cout << "Sample " << sample << std::endl;

        // Getting (image, label)
        read_image();

        // Learning process: forward_learning (Forward procedure) - Back propagation
        int nIterations = learning_process();

        // Write down the squared error
        std::cout << "Iterations: " << nIterations << std::endl;
        printf("Error: %0.6lf\n\n", square_error());
        report << "Sample " << sample << ": Iterations = " << nIterations << ", Error = " << square_error()
               << std::endl;

        // Save the current network (weights) every so often
        if (sample % 100 == 0) {
            save_weights_to_file(model_fn);
        }
    }

    // Save the final network
    save_weights_to_file(model_fn);
}

void testing()
{
    load_model_from_backup(model_fn); // Load model (weight matrices) of a trained Neural Network

    int nCorrect = 0;
    for (int sample = 1; sample <= testing_samples; ++sample) {
        std::cout << "Sample " << sample << std::endl;

        // Getting (image, label)
        int label = read_image();

        // Classification - forward_learning procedure
        forward_learning();

        // Prediction
        int predict = 1;
        for (int i = 2; i <= output_nodes; ++i) {
            if (out3[i] > out3[predict]) {
                predict = i;
            }
        }
        --predict;

        // Write down the classification result and the squared error
        float error = square_error();
        printf("Error: %0.6lf\n", error);

        if (label == predict) {
            ++nCorrect;
            std::cout << "Classification: YES. Label = " << label << ". Predict = " << predict << std::endl
                      << std::endl;
            report << "Sample " << sample << ": YES. Label = " << label << ". Predict = " << predict
                   << ". Error = " << error << std::endl;
        } else {
            std::cout << "Classification: NO.  Label = " << label << ". Predict = " << predict << std::endl;
            report << "Sample " << sample << ": NO.  Label = " << label << ". Predict = " << predict
                   << ". Error = " << error << std::endl;
        }
    }

    // Summary
    float accuracy = (float)(nCorrect) / testing_samples * 100.0;
    std::cout << "Number of correct samples: " << nCorrect << " / " << testing_samples << std::endl;
    printf("Accuracy: %0.2lf\n", accuracy);

    report << "Number of correct samples: " << nCorrect << " / " << testing_samples << std::endl;
    report << "Accuracy: " << accuracy << std::endl;
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
        image.open("mnist/train-images.idx3-ubyte", std::ios::in | std::ios::binary); // Binary image file
        label.open("mnist/train-labels.idx1-ubyte", std::ios::in | std::ios::binary); // Binary label file

        // Reading file headers
        char number;
        for (int i = 1; i <= 16; ++i) {
            image.read(&number, sizeof(char));
        }
        for (int i = 1; i <= 8; ++i) {
            label.read(&number, sizeof(char));
        }
        training();
    }
    if (will_test) {
        image.open("mnist/t10k-images.idx3-ubyte", std::ios::in | std::ios::binary); // Binary image file
        label.open("mnist/t10k-labels.idx1-ubyte", std::ios::in | std::ios::binary); // Binary label file

        // Reading file headers
        char number;
        for (int i = 1; i <= 16; ++i) {
            image.read(&number, sizeof(char));
        }
        for (int i = 1; i <= 8; ++i) {
            label.read(&number, sizeof(char));
        }
        testing();
    }

    report.close();
    image.close();
    label.close();

    return 0;
}
