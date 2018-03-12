#include "Matrix.h"
#include <math.h>

class NeuralNet{
public:

    NeuralNet(std::vector<int> shape, bool bias);

    void epoch(std::vector<std::vector<float>> inputs, std::vector<std::vector<float>> targets);
    void printReport(std::vector<std::vector<float>> inputs, std::vector<std::vector<float>> targets);

    std::shared_ptr<Matrix> feedForward(std::vector<float> inputs);
    void backPropagate(std::shared_ptr<Matrix> errors, std::vector<float> inputs);

    void printWeights();

private:

    static std::shared_ptr<Matrix> logistic(std::shared_ptr<Matrix> m){
        std::vector<std::vector<float>> out = std::vector<std::vector<float>>(m->height(), std::vector<float>(m->width()));
        for (int i = 0; i < m->height(); i++){
            for (int j = 0; j < m->width(); j++){
                out[i][j] = logistic(m->data()[i][j]);
            }
        }
        return std::make_shared<Matrix>(out);
    }

    static std::shared_ptr<Matrix> dirLogistic(std::shared_ptr<Matrix> m){
        std::vector<std::vector<float>> out = std::vector<std::vector<float>>(m->height(), std::vector<float>(m->width()));
        for (int i = 0; i < m->height(); i++){
            for (int j = 0; j < m->width(); j++){
                out[i][j] = dirLogistic(m->data()[i][j]);
            }
        }
        return std::make_shared<Matrix>(out);
    }

    static float logistic(float in){
        return (1.0 / (1.0 + std::pow(2.71828, (-in))));
    }

    static float dirLogistic(float in){
        float a = logistic(in);
        return a*(1-a);
    }

    bool _bias = false;
    std::vector<std::shared_ptr<Matrix>> _biases;
    std::vector<std::shared_ptr<Matrix>> _weights;
    std::vector<std::shared_ptr<Matrix>> _layerOutputs;

    const float LEARNINGRATE = 0.05;
    const float WEIGHTDECAY = 1;
    const float MOMENTUM = 0.9;
};
