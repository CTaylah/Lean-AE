#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(std::vector<int> topology) : m_topology(topology){
    if(m_topology.size() < 2)
        throw std::invalid_argument("Topology must have at least 2 layers");

    m_layers.push_back(Layer(0, m_topology[0]));

    for(int i = 1; i < m_topology.size(); i++){
        m_layers.push_back(Layer(m_topology[i-1], m_topology[i]));
    }
}

void NeuralNetwork::FeedForward(const Eigen::VectorXd& inputs){
    Eigen::VectorXd l_inputs = inputs;

    //Input layer has no weights or biases
    for(int i = 0; i < m_layers.size(); i++){
        l_inputs = m_layers[i].FeedForward(l_inputs);
    }
}

void NeuralNetwork::Backpropagate(const Eigen::VectorXd input, const Eigen::VectorXd& target, double learningRate, double& cost){
    FeedForward(input);
    cost = Math::MeanSquaredError(m_layers.back().GetActivations(), target);

    Layer& outputLayer = m_layers[m_layers.size() - 1];

    Eigen::VectorXd c = (outputLayer.GetActivations() - target);
    Eigen::VectorXd outputError = Math::LeakyReLUDerivative(outputLayer.GetWeightedSums()).cwiseProduct(c);

    Eigen::MatrixXd outputWeightGradients = outputError * m_layers[m_layers.size() - 2].GetActivations().transpose();
    outputWeightGradients *= learningRate;

    Eigen::VectorXd outputBiasGradients = outputError; 
    outputBiasGradients *= learningRate;
   
    outputLayer.SetWeights(outputLayer.GetWeights() - outputWeightGradients); 
    outputLayer.SetBiases(outputLayer.GetBiases() - outputBiasGradients);

    if (m_layers.size() == 2)
        return;
    
    //Calculae the error for the hidden layers
    for(int i = m_layers.size() - 2; i > 0; i--){
        Layer& hiddenLayer = m_layers[i];
        Layer& nextLayer = m_layers[i+1];

        Eigen::VectorXd hiddenError = Math::LeakyReLUDerivative(hiddenLayer.GetWeightedSums()).cwiseProduct(nextLayer.GetWeights().transpose() * outputError);

        Eigen::MatrixXd hiddenWeightGradients = hiddenError * m_layers[i-1].GetActivations().transpose();
        hiddenWeightGradients *= learningRate;

        Eigen::VectorXd hiddenBiasGradients = hiddenError;
        hiddenBiasGradients *= learningRate;

        hiddenLayer.SetWeights(hiddenLayer.GetWeights() - hiddenWeightGradients);
        hiddenLayer.SetBiases(hiddenLayer.GetBiases() - hiddenBiasGradients);

        outputError = hiddenError;
    }
    
}