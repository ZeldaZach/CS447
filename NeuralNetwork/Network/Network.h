#ifndef DEF_NETWORK
#define DEF_NETWORK

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <vector>

#include "Matrix.h"

class Network
{
public:
    Network(std::vector<int> neurons, float learningRate);
    Network(const char *filepath);

    Matrix<float> computeOutput(std::vector<float> input);
    void learn(std::vector<float> expectedOutput);

    void saveNetworkParams(const char *filepath);
    void loadNetworkParams(const char *filepath);

    std::vector<Matrix<float>> W;
    std::vector<Matrix<float>> B;

private:
    std::vector<Matrix<float>> H;
    std::vector<Matrix<float>> dEdW;
    std::vector<Matrix<float>> dEdB;

    Matrix<float> Y;

    int hiddenLayersCount;
    float learningRate;

    static float random(float x);
    static float sigmoid(float x);
    static float sigmoidePrime(float x);

    void printToFile(Matrix<float> &m, std::ofstream &file);
};

#endif
