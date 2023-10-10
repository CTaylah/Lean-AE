#pragma once

#include "Eigen/Dense"
#include <random>
#include <time.h>
#include <omp.h>


namespace Math{
    inline double ReLU(double x){
        return x > 0 ? x : 0;
    }

    inline Eigen::VectorXd ReLU(const Eigen::VectorXd& x){
        Eigen::VectorXd result(x.size());
        #pragma omp parallel for
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
        #pragma omp parallel for
        for(int i = 0; i < x.size(); i++){
            result(i) = LeakyReLU(x(i), alpha); 
        }
        return result;
    }

    inline Eigen::VectorXd LeakyReLUDerivative(const Eigen::VectorXd& x, double alpha = 0.01){
        
        Eigen::VectorXd result(x.size());
        #pragma omp parallel for
        for(int i = 0; i < x.size(); i++){
            result(i) = LeakyReLUDerivative(x(i), alpha);
        }
        return result;
    }

    inline double Sigmoid(double x){
        return 1.0 / (1.0 + exp(-x));
    }

    inline Eigen::VectorXd Sigmoid(const Eigen::VectorXd& x){
        Eigen::VectorXd result(x.size());
        #pragma omp parallel for
        for(int i = 0; i < x.size(); i++){
            result(i) = Sigmoid(x(i));
        }
        return result;
    }

    inline double SigmoidDerivative(double x){
        return Sigmoid(x) * (1.0 - Sigmoid(x));
    }

    inline Eigen::VectorXd SigmoidDerivative(const Eigen::VectorXd& x){
        Eigen::VectorXd result(x.size());
        #pragma omp parallel for
        for(int i = 0; i < x.size(); i++){
            result(i) = SigmoidDerivative(x(i));
        }
        return result;
    }

    inline double Softplus(double x){
        return log(1.0 + exp(x));
    }

    inline Eigen::VectorXd Softplus(const Eigen::VectorXd& x){
        Eigen::VectorXd result(x.size());
        #pragma omp parallel for
        for(int i = 0; i < x.size(); i++){
            result(i) = Softplus(x(i));
        }
        return result;
    }

    inline double SoftplusDerivative(double x){
        return Sigmoid(x);
    }

    inline Eigen::VectorXd SoftplusDerivative(const Eigen::VectorXd& x){
        Eigen::VectorXd result(x.size());
        #pragma omp parallel for
        for(int i = 0; i < x.size(); i++){
            result(i) = SoftplusDerivative(x(i));
        }
        return result;
    }
    
    inline double UndoLog(double x){
        return exp(x);
    }   

    inline Eigen::VectorXd UndoLog(const Eigen::VectorXd& x){
        Eigen::VectorXd result(x.size());
        #pragma omp parallel for
        for(int i = 0; i < x.size(); i++){
            result(i) = UndoLog(x(i));
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

    inline double KL(const Eigen::VectorXd& mu_q, const Eigen::VectorXd& logvar_q,
        const Eigen::VectorXd& mu_p, const Eigen::VectorXd& logvar_p) 
    {
        Eigen::VectorXd var_q = logvar_q.array().exp();
        Eigen::VectorXd var_p = logvar_p.array().exp();
        
        Eigen::VectorXd logvar_q_inv = logvar_q.array().exp().cwiseInverse();
        Eigen::VectorXd logvar_p_inv = logvar_p.array().exp().cwiseInverse();
        
        Eigen::VectorXd logvar_p_q = logvar_p - logvar_q;
        Eigen::VectorXd mu_p_mu_q = mu_p - mu_q;
        
        Eigen::VectorXd mu_p_mu_q_sq = mu_p_mu_q.array().square();
        Eigen::VectorXd var_p_q = var_p.cwiseQuotient(var_q);
        Eigen::VectorXd var_p_q_sq = var_p_q.array().square();
        
        double kl = 0.5 * (logvar_p_q.sum() + var_p_q_sq.sum() + mu_p_mu_q_sq.sum() - mu_p.size());
        
        return kl / logvar_p.size();
    }
    

    inline void ShuffleMatrix(Eigen::Ref<Eigen::MatrixXd>& matrix){
        for(int i = 0; i < matrix.cols(); i++){
            int randomIndex = rand() % matrix.cols();
            matrix.col(i).swap(matrix.col(randomIndex));
        }
    }

    inline Eigen::VectorXd genGaussianVector(int size, double mean, double stddev)
    {
        Eigen::VectorXd samples(size);
        
        std::random_device rd; 
        std::mt19937 gen(rd());
        
        std::normal_distribution<double> distribution(mean, stddev);
        
        for (int i = 0; i < size; i++) {
            samples(i) = distribution(gen);
        }
        
        return samples;
    }
   
}


