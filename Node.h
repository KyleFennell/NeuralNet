#pragma once

#include "Math.h"
#include <vector>
#include <cstdlib>

class Node{
public:

    Node(int inputCount);
    ~Node();

    void calculate(std::vector<float> intputs);
    float output(){ return _output; }
    void backProporgate(float error, float output);
    float error(){ return _error; }

private:

    std::vector<float> _weights;
    float _output;
    float _error;

};
