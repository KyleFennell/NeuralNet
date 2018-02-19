#pragma once

#include "Node.h"

class Layer{
public:

    Layer(int layerSize, int prevLayerSize);
    ~Layer();

    void calculate(std::vector<float> inputs);
    std::vector<float> output(){ return _output; }
    void backProporgate(std::vector<float> error, std::vector<float> output);
    std::vector<float> error(){ return _error; }

private:

    std::vector<Node*> _nodes;
    std::vector<float> _output;
    std::vector<float> _error;

};
