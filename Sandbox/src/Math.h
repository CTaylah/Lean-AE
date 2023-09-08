#pragma once

#include <random>

namespace Math{
    inline double RandomDouble(double min, double max){
        std::uniform_real_distribution<double> uniformReal(min,max);
        std::default_random_engine randomEngine;
        return uniformReal(randomEngine);
    }

    inline int RandomInt(int min, int max){
        std::uniform_int_distribution<int> uniformInt(min,max);
        std::default_random_engine randomEngine;
        return uniformInt(randomEngine);
    }

    inline double ReLU(double x){
        return x > 0 ? x : 0;
    }

    inline double ReLUDerivative(double x){
        return x > 0 ? 1 : 0;
    }
}


