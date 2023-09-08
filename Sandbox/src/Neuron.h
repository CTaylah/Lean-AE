#pragma once
#include "Math.hpp"

#include "Eigen/Dense"


class Neuron{
    public:
        Neuron(int numInputs) {
            m_weights = Eigen::VectorXd::Random(numInputs);
            m_bias = Math::RandomDouble(-0.1, 0.1);
        }

        double GetActivation(Eigen::VectorXd inputs);

        double GetActivationDerivative(Eigen::VectorXd inputs);
        

        Eigen::VectorXd GetWeights(){ return m_weights; }
        Eigen::VectorXd& GetWeightsRef(){ return m_weights; }
        double GetBias(){return m_bias; } 


    private:
        Eigen::VectorXd m_weights;
        double m_bias;
 
};