#pragma once

#include "Layer.h"

class NeuralNet{
public:

    NeuralNet(std::vector<int> layerStructure);
    void calculate(std::vector<float> inputs);
    std::vector<float> output(){ return _output; }
    void backProporgate(std::vector<float> error);

private:

    std::vector<Layer*> _layers;
    std::vector<float> _output;

};
