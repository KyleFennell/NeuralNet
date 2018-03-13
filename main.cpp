#include "NeuralNet.h"

int main(int argc, char** argv){

    std::vector<arma::mat> inputs = {   {0, 0, 0},
                                        {0, 1, 0},
                                        {1, 0, 0},
                                        {1, 1, 0},
                                        {0, 0, 1},
                                        {0, 1, 1},
                                        {1, 0, 1},
                                        {1, 1, 1}};
    std::vector<arma::mat> targets = {  {0},
                                        {1},
                                        {1},
                                        {0},
                                        {1},
                                        {0},
                                        {0},
                                        {1}};

//    std::vector<arma::mat> inputs = {  {0, 0},
//                                                {0, 1},
//                                                {1, 0},
//                                                {1, 1}};
//    std::vector<arma::mat> targets = { {0},
//                                                {1},
//                                                {1},
//                                                {0}};

    std::unique_ptr<NeuralNet> nn = std::make_unique<NeuralNet>(std::vector<int>({3, 7, 1}), true);
//    nn->printReport(inputs, targets);
    for (int i = 0; i < 5000000; i++){
        if (i%10000 == 0){
            std::cout << "iteration: " << i << std::endl;
            nn->printReport(inputs, targets);
        }
        if(nn->epoch(inputs, targets) < 0.05){
            std::cout << "Finished on itteration: " << i << std::endl;
            break;
        }
    }
    nn->printReport(inputs, targets);
    return 0;
}
