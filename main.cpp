#include "NeuralNet.h"

int main(int argc, char* argv[]){

    std::vector<int> netSize = {2, 1, 1};
    NeuralNet net(netSize);
    std::vector<std::vector<float>> indataset = {  {0, 0},
                                    {0, 1},
                                    {1, 0},
                                    {1, 1}};
    std::vector<float> outdataset = {0, 0, 0, 1};
    for (int j = 0; j < 5000; j++){
        for (int i = 0; i < (int)indataset.size(); i++){
            net.calculate(indataset[i]);
            float error = -(net.output()[0]-outdataset[i]);
            std::vector<float> err = {error};
            net.backProporgate(err);
        }
        if (j % 50 == 0){
            std::cout << "inputs   output" << std::endl;
            for (int i = 0; i < (int)indataset.size(); i++){
                net.calculate(indataset[i]);
                std::cout << indataset[i][0] << " " << indataset[i][1] << " " << net.output()[0] << " " << net.output()[0]-outdataset[i] << std::endl;
            }
        }
    }
    return 0;
}
