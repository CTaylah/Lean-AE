#include "Layer.h"

Eigen::VectorXd Layer::FeedForward(const Eigen::VectorXd& inputs){
    //If this is the input layer, just return the inputs
    if(m_weights.size() == 0){
        m_activations = inputs;
        return inputs;
    }

    if(inputs.size() != m_weights.cols()){
        throw std::invalid_argument("Layer::FeedForward: invalid number of inputs: size:");
    }

    m_weightedSums = (m_weights * inputs) + m_biases;
    m_activations = Math::LeakyReLU(m_weightedSums);


    return m_activations;    

}