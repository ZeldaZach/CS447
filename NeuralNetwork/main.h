//
// Created by ZaHal on 5/2/2019.
//

#ifndef CS447_MAIN_H
#define CS447_MAIN_H

#include <string>

class NeuralNetwork
{
public:
    template <int N> void read_mnist_labels(const std::string &fn, unsigned char (&labels)[N]);

    template <int N> void read_mnist_images(const std::string &fn, float (&imgs)[N][28][28]);

    int read_int(int fd);

    void swap(int &i);

    //__global__ train(float *training_images_cuda, float *training_labels_cuda);

    /*void read_input_data(float training_images[][28][28],
                         unsigned char training_labels[],
                         float test_images[][28][28],
                         unsigned char test_labels[]);*/
};
#endif // CS447_MAIN_H
