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

    switch(m_activationFunction){
        case ActivationFunction::SIGMOID:
            m_activations = Math::Sigmoid(m_weightedSums);
            break;
        case ActivationFunction::RELU:
            m_activations = Math::ReLU(m_weightedSums);
            break;
        case ActivationFunction::LEAKY_RELU:
            m_activations = Math::LeakyReLU(m_weightedSums);
            break;
        case ActivationFunction::SOFTPLUS:
            m_activations = Math::Softplus(m_weightedSums);
            break;
        default:
            throw std::invalid_argument("Layer::FeedForward: invalid activation function");
    }

    return m_activations;    

}