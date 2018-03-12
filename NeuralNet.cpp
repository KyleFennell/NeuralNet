#include "NeuralNet.h"
#include <time.h>

NeuralNet::NeuralNet(std::vector<int> shape, bool bias){
    srand(time(NULL));
    for (int i = 0; i < (int)shape.size()-1; i++){
        _weights.push_back(std::make_shared<Matrix>(shape[i+1], shape[i]));
        _bias = bias;
        std::shared_ptr<Matrix> tempBias = std::make_shared<Matrix>(shape[i+1], 1);
        if (!_bias){
            tempBias = Matrix::multiply(tempBias, 0);       //sets all bias to 0 so they wont interfer with feedforward.
        }
//        std::cout << *tempBias << std::endl;
        _biases.push_back(tempBias);
    }
}

void NeuralNet::printWeights(){
    for (int i = 0; i < (int)_weights.size(); i++){
        std::cout << *_weights[i] << std::endl;
    }
}

void NeuralNet::epoch(std::vector<std::vector<float>>inputs, std::vector<std::vector<float>>targets){
    int order[inputs.size()];
    int count = 0;
    while(count < (int)inputs.size()){               // randomising training order to prevent clashes in learning
        int num = rand()%inputs.size();
        bool found = false;
        for (int i = 0; i < count && !found; i++){
            if (order[i] == num){
                found = true;
            }
        }
        if (!found){
            order[count] = num;
            count++;
        }
    }
    for(int i = 0; i < (int)inputs.size(); i++){
        std::shared_ptr<Matrix> output = feedForward(inputs[order[i]]);
//        std::shared_ptr<Matrix> error = Matrix(targets[order[i]]).add(output->multiply(-1));        std::shared_ptr<Matrix> error = output->add(Matrix::multiply(std::make_shared<Matrix>(targets[order[i]]), -1));
        backPropagate(error, inputs[order[i]]);
    }
}

void NeuralNet::printReport(std::vector<std::vector<float>>inputs, std::vector<std::vector<float>>targets){
    const std::string floatLength = "         ";
    std::string out = "";
    out += "input    "; for (int i = 0; i < Matrix(inputs[0]).width()-1; i++) {out += floatLength;}
    out += "target   "; for (int i = 0; i < Matrix(targets[0]).width()-1; i++) {out += floatLength;}
    out += "output   "; for (int i = 0; i < Matrix(targets[0]).width()-1; i++) {out += floatLength;}
    out += "errors   "; for (int i = 0; i < Matrix(targets[0]).width()-1; i++) {out += floatLength;}
    for(int i = 0; i < (int)inputs.size(); i++){
        std::shared_ptr<Matrix> output = feedForward(inputs[i]);
        std::shared_ptr<Matrix> error = Matrix(targets[i]).add(output->multiply(-1));
        out += "\n" + std::make_unique<Matrix>(inputs[i])->to_string() + std::make_unique<Matrix>(targets[i])->to_string() + output->to_string() + error->to_string();
    }
    std::cout << out << std::endl;
}

std::shared_ptr<Matrix> NeuralNet::feedForward(std::vector<float> inputs){
//    std::cout << "\tFEEDFORWARD" << std::endl;
    _layerOutputs.clear();
//    std::cout << "\ninputs: " << Matrix(inputs).size() << "\nDOT weights: " << _weights[0]->size() << "\nEQUALS output: " << Matrix(inputs).dot(_weights[0])->size() << "\nADD biases: " << _biases[0]->size() << "\nEQUALS output: " << Matrix::add(Matrix(inputs).dot(_weights[0]), _biases[0])->size() << "\nlogistic performed" << std::endl;
    _layerOutputs.push_back(logistic(Matrix::add(Matrix(inputs).dot(_weights[0]), _biases[0])));
    for (int i = 1; i < (int)_weights.size(); i++){
//        std::cout << "\ninputs: " << _layerOutputs[i-1]->size() << "\nDOT weights: " << _weights[i]->size() << "\nEQUALS 0utputs: " << _layerOutputs[i-1]->dot(_weights[i])->size() << "\nADD biases: " << _biases[i]->size() << "\nEQUALS output: " << Matrix::add(_layerOutputs[i-1]->dot(_weights[i]), _biases[i])->size() << "\nlogistic performed" << std::endl;
        _layerOutputs.push_back(logistic(Matrix::add(_layerOutputs[i-1]->dot(_weights[i]), _biases[i])));
    }
    return _layerOutputs.back();
}

void NeuralNet::backPropagate(std::shared_ptr<Matrix> errors, std::vector<float> inputs){
//    std::cout << "\tBACKPROP" << std::endl;
    std::shared_ptr<Matrix> weightDeltas = errors;
    weightDeltas = Matrix::dot(weightDeltas, dirLogistic(_layerOutputs.back()));
    for (int i = _layerOutputs.size()-1; i > 0; i--){
//        std::cout << "\nlayerOut.t: " << _layerOutputs[i-1]->t()->size() << "\nDOT weightDeltas: " << weightDeltas->size() << "\nEQUALS : " << Matrix::dot(_layerOutputs[i-1]->t(), weightDeltas)->size() << "\nADD weights: " << _weights[i]->size() << std::endl;
        _weights[i] = Matrix::multiply(Matrix::add(_weights[i], Matrix::multiply(Matrix::dot(_layerOutputs[i-1]->t(), weightDeltas), -LEARNINGRATE)), WEIGHTDECAY);
        if (_bias){
//            std::cout << "\nweightDeltas: " << weightDeltas->size() << "\nADD weights: " << _biases[i]->size() << std::endl;
            _biases[i] = Matrix::multiply(
                Matrix::add(_biases[i], Matrix::multiply(weightDeltas, -LEARNINGRATE)), WEIGHTDECAY);
        }
//        std::cout << "\noldWeightDeltas: " << weightDeltas->size() << "\nDOT weights.t: " << _weights[i]->t()->size() << "\nEQUALS : " << Matrix::dot(weightDeltas, _weights[i]->t())->size() << "\nMULTIPLY layerOut: " << _layerOutputs[i-1]->size() << std::endl;
        weightDeltas = Matrix::multiply(Matrix::dot(weightDeltas, _weights[i]->t()), dirLogistic(_layerOutputs[i-1]));
    }
//    std::cout << "\nlayerOut.t: " << std::make_unique<Matrix>(inputs)->t()->size() << "\nDOT weightDeltas: " << weightDeltas->size() << "\nEQUALS : " << Matrix::dot(std::make_unique<Matrix>(inputs)->t(), weightDeltas)->size() << "\nADD weights: " << _weights[0]->size() << std::endl;    if (_bias){
//        std::cout << "\nweightDeltas: " << weightDeltas->size() << "\nADD weights: " << _biases[0]->size() << std::endl;
        _biases[0] = Matrix::multiply(Matrix::add(_biases[0], Matrix::multiply(weightDeltas, -LEARNINGRATE)), WEIGHTDECAY);
    }
}
