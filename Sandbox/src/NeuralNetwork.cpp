#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(Topology topology) : m_topology(topology){

    m_layers.push_back(Layer(0, m_topology.inputSize));

    for(int i = 1; i < m_topology.neuronsPerLayer.size(); i++){
        m_layers.push_back(Layer(m_topology.neuronsPerLayer[i-1], m_topology.neuronsPerLayer[i]));
    }
}

void NeuralNetwork::FeedForward(const Eigen::VectorXd& inputs){
    Eigen::VectorXd l_inputs = inputs;

    //Input layer has no weights or biases
    for(int i = 1; i < m_layers.size(); i++){
        l_inputs = m_layers[i].FeedForward(l_inputs);
    }
}

void NeuralNetwork::Backpropagate(const Eigen::VectorXd& target, double learningRate){
    //First, calculate the error for the output layer
    Layer& outputLayer = m_layers[m_layers.size() - 1];
    Eigen::VectorXd outputError = Math::ReLUDerivative(outputLayer.GetWeightedSums()) *(2 * outputLayer.GetActivations() - target);

    Eigen::MatrixXd outputWeightGradients = outputError * m_layers[m_layers.size() - 2].GetActivations().transpose();
    outputWeightGradients *= learningRate;

    Eigen::VectorXd outputBiasGradients = outputError; 
    outputBiasGradients *= learningRate;

    //Update the weights and biases for the output layer
    outputLayer.SetWeights(outputLayer.GetWeights() - outputWeightGradients); 
    outputLayer.SetBiases(outputLayer.GetBiases() - outputBiasGradients);

    //Calculae the error for the hidden layers
    for(int i = m_layers.size() - 2; i > 0; i--){
        Layer& hiddenLayer = m_layers[i];
        Layer& nextLayer = m_layers[i+1];

        Eigen::VectorXd hiddenError = Math::ReLUDerivative(hiddenLayer.GetWeightedSums()) * (nextLayer.GetWeights().transpose() * outputError);

        Eigen::MatrixXd hiddenWeightGradients = hiddenError * m_layers[i-1].GetActivations().transpose();
        hiddenWeightGradients *= learningRate;

        Eigen::VectorXd hiddenBiasGradients = hiddenError;
        hiddenBiasGradients *= learningRate;

        //Update the weights and biases for the hidden layer
        hiddenLayer.SetWeights(hiddenLayer.GetWeights() - hiddenWeightGradients);
        hiddenLayer.SetBiases(hiddenLayer.GetBiases() - hiddenBiasGradients);

        outputError = hiddenError;
    }
    
}