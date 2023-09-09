#include "Layer.h"

Eigen::VectorXd Layer::FeedForward(const Eigen::VectorXd& inputs){
    if(inputs.size() != m_weights.cols()){
        throw std::invalid_argument("Layer::FeedForward: invalid number of inputs");
    }

    m_weightedSums = (m_weights * inputs) + m_biases;

    for(int i = 0; i < m_activations.rows(); i++){
        m_activations(i) = Math::ReLU(m_weightedSums(i));
    }

    return m_activations;    

}