#include "Node.h"

Node::Node(int inputCount){
//    std::cout << "creating new node" << std::endl;
    for (int i = 0 ; i <= inputCount; i++){
        _weights.push_back((float)rand() / RAND_MAX);
//        std::cout << "adding weight: " << _weights[i] << std::endl;
    }
}

Node::~Node(){}

void Node::calculate(std::vector<float> inputs){
    inputs.push_back(1);    // bias
    std::vector<float> weightedInputs = Math::multiply(inputs, _weights);
    _output = Math::activate(Math::sum(weightedInputs));
}

void Node::backProporgate(float error, float output){
    _error = 0;
    for (int i = 0; i < (int)_weights.size(); i++){
        float dWeight = error*Math::dActivate(output);
        _error += dWeight;
        _weights[i] -= dWeight;
    }
}
