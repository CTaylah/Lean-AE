#include "Layer.h"


Eigen::VectorXd Layer::FeedForward(const Eigen::VectorXd& inputs){
    //If this is the input layer, just return the inputs
    if(m_weights.size() == 0){
        m_activations = inputs;
        return inputs;
    }

    if(inputs.size() != m_weights.cols()){
        std::cout << inputs.size() << std::endl;
        std::cout << m_weights.cols() << std::endl;
        throw std::invalid_argument("Layer::FeedForward: invalid number of inputs: size:");
    }

    m_weightedSums = (m_weights * inputs) + m_biases;
    m_activations = CalculateActivation(m_weightedSums);
    return m_activations;    
}

Eigen::VectorXd Layer::CalculateActivation(const Eigen::VectorXd& weightedSums){
    switch(m_activationFunction){
        case ActivationFunction::SIGMOID:
            return Math::Sigmoid(weightedSums);
        case ActivationFunction::RELU:
            return Math::ReLU(weightedSums);
        case ActivationFunction::LEAKY_RELU:
            return Math::LeakyReLU(weightedSums);
        case ActivationFunction::SOFTPLUS:
            return Math::Softplus(weightedSums);
        default:
            throw std::invalid_argument("Layer::CalculateActivation: invalid activation function");
    }
}

Eigen::VectorXd Layer::ActivationDerivative(const Eigen::VectorXd& activations){
    switch(m_activationFunction){
        case ActivationFunction::SIGMOID:
            return Math::SigmoidDerivative(activations);
        case ActivationFunction::RELU:
            return Math::ReLUDerivative(activations);
        case ActivationFunction::LEAKY_RELU:
            return Math::LeakyReLUDerivative(activations);
        case ActivationFunction::SOFTPLUS:
            return Math::SoftplusDerivative(activations);
        default:
            throw std::invalid_argument("Layer::ActivationDerivative: invalid activation function");
    }
}