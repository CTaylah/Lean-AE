#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(Topology topology) : m_topology(topology){

    m_layers.push_back(Layer(0, m_topology.inputSize));

    for(int i = 1; i < m_topology.neuronsPerLayer.size(); i++){
        m_layers.push_back(Layer(m_topology.neuronsPerLayer[i-1], m_topology.neuronsPerLayer[i]));
    }

}

void NeuralNetwork::FeedForward(const Eigen::VectorXd& inputs){
    for(int i = 1; i < m_layers.size(); i++){
    }
}