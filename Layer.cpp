#include "Layer.h"

Layer::Layer(int layerSize, int prevLayerSize){
//    std::cout << "new layer of size: " << layerSize << std::endl;
    _nodes = std::vector<Node*>(layerSize, new Node(prevLayerSize));
//    std::cout << "layer created" << std::endl;
}

void Layer::calculate(std::vector<float> inputs){
    _output.clear();
    for (int i = 0; i < (int)_nodes.size(); i++){
        _nodes[i]->calculate(inputs);
        _output.push_back(_nodes[i]->output());
    }
}

void Layer::backProporgate(std::vector<float> error, std::vector<float> output){
    _error.clear();
    for(int i = 0; i < (int)error.size(); i++){
        _nodes[i]->backProporgate(error[i], output);
        _error.push_back(_nodes[i]->error());
    }

}
