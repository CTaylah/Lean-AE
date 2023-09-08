#include "Neuron.h"


double Neuron::GetActivation(Eigen::VectorXd inputs){
    return Math::ReLU(m_weights.dot(inputs) + m_bias);
}

double Neuron::GetActivationDerivative(Eigen::VectorXd inputs){
    return Math::ReLUDerivative(m_weights.dot(inputs) + m_bias);
}