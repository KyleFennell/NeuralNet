#pragma once

#include <vector>
#include <cmath>
#include <iostream>

class Math{
public:

    static float activate(float in){
        return (1 / (1 + std::pow(2.178, -in)));
    }

    static float dActivate(float in){
        float a = activate(in);
        return a*(1-a);
    }

    static std::vector<float> multiply(std::vector<float> a, std::vector<float> b){
        std::vector<float> out;
        for (int i = 0; i < (int)a.size(); i++){
            out.push_back(a[i] * b[i]);
        }
        return out;
    }

    static float sum(std::vector<float> in){
        float sum = 0;
        for (int i = 0; i < (int)in.size(); i++){
            sum += in[i];
        }
        return sum;
    }

};
