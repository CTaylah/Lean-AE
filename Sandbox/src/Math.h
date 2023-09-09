#pragma once

#include "Eigen/Dense"
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

    inline Eigen::VectorXd ReLU(const Eigen::VectorXd& x){
        Eigen::VectorXd result(x.size());
        for(int i = 0; i < x.size(); i++){
            result(i) = ReLU(x(i));
        }
        return result;
    }

    inline double ReLUDerivative(double x){
        return x > 0 ? 1 : 0;
    }

    inline Eigen::VectorXd ReLUDerivative(const Eigen::VectorXd& x){
        Eigen::VectorXd result(x.size());
        for(int i = 0; i < x.size(); i++){
            result(i) = ReLUDerivative(x(i));
        }
        return result;
    }

    inline double SquaredError(Eigen::VectorXd& prediction, Eigen::VectorXd& label){
        return (prediction - label).squaredNorm();
    }
    
}


