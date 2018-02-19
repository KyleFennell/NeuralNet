#include "NeuralNet.h"

NeuralNet::NeuralNet(std::vector<int> layerSize){
//    std::cout << "creating layer: " << 0 << " size: " << layerSize[0] << std::endl;
    _layers.push_back(new Layer(layerSize[0], 1));
    for (int i = 1; i < (int)layerSize.size(); i++){
//        std::cout << "creating layer: " << i << " size: " << layerSize[i] << std::endl;
        _layers.push_back(new Layer(layerSize[i], layerSize[i-1]));
    }
}

void NeuralNet::calculate(std::vector<float> input){
    _layers[0]->calculate(input);
    for(int i = 1; i < (int)_layers.size(); i++){
        _layers[i]->calculate(_layers[0]->output());
    }
    _output = _layers[_layers.size()-1]->output();
}

void NeuralNet::backProporgate(std::vector<float> error){
    _layers[_layers.size()-1]->backProporgate(error, _layers[_layers.size()-2]->output());
    for (int i = _layers.size()-2; i >= 0; i--){
        _layers[i]->backProporgate(error, _layers[i+1]->output());
    }
}
