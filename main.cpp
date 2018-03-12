#include "NeuralNet.h"

int main(int argc, char** argv){

    std::vector<std::vector<float>> inputs = {  {0, 0, 0},
                                                {0, 1, 0},
                                                {1, 0, 0},
                                                {1, 1, 0},
                                                {0, 0, 1},
                                                {0, 1, 1},
                                                {1, 0, 1},
                                                {1, 1, 1}};
    std::vector<std::vector<float>> targets = { {0},
                                                {1},
                                                {1},
                                                {0},
                                                {1},
                                                {0},
                                                {0},
                                                {1}};

//    std::vector<std::vector<float>> inputs = {  {0, 0},
//                                                {0, 1},
//                                                {1, 0},
//                                                {1, 1}};
//    std::vector<std::vector<float>> targets = { {0},
//                                                {1},
//                                                {1},
//                                                {0}};

    std::unique_ptr<NeuralNet> nn = std::make_unique<NeuralNet>(std::vector<int>({3, 5, 5, 1}), false);
    nn->printReport(inputs, targets);
    for (int i = 0; i < 500000; i++){
//        std::cout << "\tSTARTING EPOCH" << std::endl;
        nn->epoch(inputs, targets);
        if (i%1000 == 0){
            std::cout << i << std::endl;
            nn->printReport(inputs, targets);
        }
//        std::cout << "\tFINISHED EPOCH" << std::endl;
    }
    nn->printReport(inputs, targets);
    return 0;
}
