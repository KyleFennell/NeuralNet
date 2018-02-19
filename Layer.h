#pragma once

#include "Node.h"

class Layer{
public:

    Layer(int layerSize, int prevLayerSize);
    ~Layer();

    void calculate(std::vector<float> inputs);
    std::vector<float> output(){ return _output; }
    void backProporgate(std::vector<float> errors, std::vector<float> outputs);
    std::vector<float> errors(){ return _errors; }

private:

    std::vector<Node*> _nodes;
    std::vector<float> _output;
    std::vector<float> _errors;

};
