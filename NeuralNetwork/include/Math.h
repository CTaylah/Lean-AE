#pragma once

#include "Eigen/Dense"
#include <random>
#include <time.h>


namespace Math{
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

    inline double LeakyReLU(double x, double alpha = 0.01){
        return x > 0 ? x : alpha * x;
    }

    inline double LeakyReLUDerivative(double x, double alpha = 0.01){
        return x > 0 ? 1 : alpha;
    }

    inline Eigen::VectorXd LeakyReLU(const Eigen::VectorXd& x, double alpha = 0.01){
        Eigen::VectorXd result(x.size());
        for(int i = 0; i < x.size(); i++){
            result(i) = LeakyReLU(x(i), alpha); 
        }
        return result;
    }

    inline Eigen::VectorXd LeakyReLUDerivative(const Eigen::VectorXd& x, double alpha = 0.01){
        
        Eigen::VectorXd result(x.size());
        for(int i = 0; i < x.size(); i++){
            result(i) = LeakyReLUDerivative(x(i), alpha);
        }
        return result;
    }

    inline double SquaredError(const Eigen::VectorXd& prediction, const Eigen::VectorXd& label){
        return 0.5 * (prediction - label).squaredNorm();
    }

    inline Eigen::VectorXd SquaredErrorDerivative(const Eigen::VectorXd& prediction, const Eigen::VectorXd& label){
        return (prediction - label);
    }

    inline double MeanSquaredError(const Eigen::VectorXd& prediction, const Eigen::VectorXd& label){
        return 0.5 * ((prediction - label).squaredNorm()) / prediction.size();
    }

    inline void ShuffleMatrix(Eigen::Ref<Eigen::MatrixXd>& matrix){
        for(int i = 0; i < matrix.cols(); i++){
            int randomIndex = rand() % matrix.cols();
            matrix.col(i).swap(matrix.col(randomIndex));
        }
    }
   
}


