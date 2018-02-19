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

void Node::backProporgate(float errorOfPrev, std::vector<float> outputOfNext){
    float sumErrorOfThis = 0;
    for (int i = 0; i < (int)_weights.size(); i++){
        float error = Math::dActivate(_output)*_weights[i];       //dh
        sumErrorOfThis += error;                                  // sum dh
    }
    _error = sumErrorOfThis*errorOfPrev;                 // dg
    for (int i = 0; i < (int)outputOfNext.size(); i++){
        float dWeight = errorOfPrev*Math::dActivate(_output)*outputOfNext[i];       //dh
        _weights[i] -= 0.01*dWeight;                                          // sum dh
    }
    _weights[_weights.size()-1] -= 0.01*errorOfPrev*Math::dActivate(_output);
}
