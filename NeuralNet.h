#include <armadillo>        //almost everything (matrecies)
#include <math.h>           //pow
#include <memory>           //smart pointers

class NeuralNet{
public:

    NeuralNet(std::vector<int> shape, bool bias);

    float epoch(std::vector<arma::mat> inputs, std::vector<arma::mat> targets);
    void printReport(std::vector<arma::mat>inputs, std::vector<arma::mat>targets);

    arma::mat feedForward(arma::mat inputs);
    void backPropagate(arma::mat errors, arma::mat inputs);

    void printWeights();

private:

    static arma::mat logistic(arma::mat m){
        return m.for_each( [](arma::mat::elem_type& val) { val = logistic(val); } );
    }

    static arma::mat dirLogistic(arma::mat m){
        return m.for_each( [](arma::mat::elem_type& val) { val = dirLogistic(val); } );
    }

    static arma::mat addConst(arma::mat m, float add){
        return m.for_each( [add](arma::mat::elem_type& val) { val = val + add; } );
    }

    static arma::mat multConst(arma::mat m, float mult){
        return m.for_each( [mult](arma::mat::elem_type& val) { val = val * mult; } );
    }

    static float logistic(float in){
        return (1.0 / (1.0 + std::pow(2.71828, (-in))));
    }

    static float dirLogistic(float in){
        float a = logistic(in);
        return a*(1-a);
    }

    static void size(arma::mat m){
        std::cout << "(" << m.n_cols << "," << m.n_rows << ")" << std::endl;
    }

    bool _bias = false;
    std::vector<arma::mat> _biases;
    std::vector<arma::mat> _weights;
    std::vector<arma::mat> _layerOutputs;

    constexpr static float LEARNINGRATE = 0.06;
};
