#include "NeuralNet.h"
#include <time.h>

NeuralNet::NeuralNet(std::vector<int> shape, bool bias){
    srand(time(NULL));
    _bias = bias;
    for (int i = 0; i < (int)shape.size()-1; i++){
        arma::mat tempWeight(shape[i+1], shape[i], arma::fill::randu);
        tempWeight.for_each( [](arma::mat::elem_type& val) {val = val*2-1;});
        _weights.push_back(tempWeight);
        arma::mat tempBias(shape[i+1], 1, arma::fill::randu);
        tempBias.for_each( [](arma::mat::elem_type& val) {val = val*2-1;});
        if (!_bias){
            tempBias.zeros();       //sets all bias to 0 so they wont interfer with feedforward.
        }
//        std::cout << *tempBias << std::endl;
        _biases.push_back(tempBias);
    }
}

float NeuralNet::epoch(std::vector<arma::mat> inputs, std::vector<arma::mat> targets){
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
    float errorsum = 0;
    for(int i = 0; i < (int)inputs.size(); i++){
        arma::mat output = feedForward(inputs[order[i]]);
//        std::shared_ptr<Matrix> error = Matrix(targets[order[i]]).add(output->multiply(-1));//        std::shared_ptr<Matrix> error = output->add(Matrix::multiply(std::make_shared<Matrix>(targets[order[i]]), -1));
        arma::mat error = output - targets[order[i]];
        for (int i = 0; i < (int)error.n_cols; i++)
            errorsum += error[i] * error[i];
        backPropagate(error, inputs[order[i]]);
    }
    return std::pow(errorsum/(inputs.size()*targets[0].n_cols), 0.5);
}

void NeuralNet::printReport(std::vector<arma::mat>inputs, std::vector<arma::mat>targets){
    const std::string floatLength = "         ";
    std::string out = "";
    out += "input  "; for (int i = 7; i < (int)inputs[0].n_cols-1; i++) {out += "  ";}
    out += "target  "; for (int i = 8; i < (int)targets[0].n_cols-1; i++) {out += "  ";}
    out += "output   "; for (int i = 0; i < (int)targets[0].n_cols-1; i++) {out += floatLength;}
    out += "\terrors   "; for (int i = 0; i < (int)targets[0].n_cols-1; i++) {out += floatLength;}
    std::cout << out << std::endl;
    float errorsum = 0;
    for(int i = 0; i < (int)inputs.size(); i++){
        arma::mat output = feedForward(inputs[i]);
        arma::mat error = output - targets[i];
        for (int j = 0; j < (int)inputs[0].n_cols; j++) {std::cout << inputs[i][j] << " ";}
        for (int j = inputs[0].n_cols*2; j < 7; j++) {std::cout << " ";}
        for (int j = 0; j < (int)targets[0].n_cols; j++) {std::cout << targets[i][j] << " ";}
        for (int j = targets[0].n_cols*2; j < 8; j++) {std::cout << " ";}
        for (int j = 0; j < (int)targets[0].n_cols; j++) {std::cout << output[j] << " ";}
        std::cout << "  \t";
        for (int j = 0; j < (int)targets[0].n_cols; j++) {std::cout << error[j] << " ";}
        std::cout << std::endl;
        for (int i = 0; i < (int)error.n_cols; i++)
            errorsum += error[i] * error[i];
//        out += "\n" + inputs + targets + output + error;
    }
    std::cout << "RMSE: " << std::pow(errorsum/(inputs.size()*targets[0].n_cols), 0.5) << std::endl;
}

arma::mat NeuralNet::feedForward(arma::mat inputs){
//        std::cout << "\tFEEDFORWARD" << std::endl;
    _layerOutputs.clear();
//        size(inputs);//        size(_weights[0]);//        size(inputs.t());
//        size(_weights[0].t());
//        size(_biases[0]);
    _layerOutputs.push_back(logistic(_biases[0] + (_weights[0] * inputs.t())));
    for (int i = 1; i < (int)_weights.size(); i++){
        _layerOutputs.push_back(logistic(_biases[i] + (_weights[i] * _layerOutputs[i-1])));
    }
    return _layerOutputs.back();
}

void NeuralNet::backPropagate(arma::mat errors, arma::mat inputs){
//        std::cout << "\tBACKPROP" << std::endl;
    arma::mat weightDeltas = errors;
//        size(_layerOutputs.back());
//        size(weightDeltas);
    weightDeltas = weightDeltas * dirLogistic(_layerOutputs.back());
    for (int i = _layerOutputs.size()-1; i > 0; i--){
//            std::cout << "weights " << i << std::endl;
//            size(weightDeltas);
//            size(weightDeltas.t());
//            size(_layerOutputs[i-1]);
//            size(_layerOutputs[i-1].t());//            size(_weights[i]);
//            std::cout << std::endl;
        _weights[i] = _weights[i] + NeuralNet::multConst((weightDeltas * _layerOutputs[i-1].t()), -LEARNINGRATE);
        if (_bias){
//               std::cout << "biases " << i << std::endl;
//               size(weightDeltas);
//               size(_biases[i]);
//               std::cout << std::endl;
            _biases[i] = _biases[i] + NeuralNet::multConst(weightDeltas, -LEARNINGRATE);
        }
//            std::cout << "deltas " << std::endl;
//            size(weightDeltas);
//            size(_weights[i].t());
//            size(_layerOutputs[i-1]);
        weightDeltas = (_weights[i].t() * weightDeltas) % dirLogistic(_layerOutputs[i-1]);
    }
        _weights[0] = _weights[0] + NeuralNet::multConst((weightDeltas * inputs), -LEARNINGRATE);
    if (_bias){
        _biases[0] = _biases[0] + multConst(weightDeltas, -LEARNINGRATE);
    }
}
