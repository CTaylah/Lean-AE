#include "NeuralNetwork.h"
#include <Eigen/StdVector>
#include <omp.h>

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

void NeuralNetwork::BackpropagateBatch(const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& targets, double learningRate, double& cost){
    if(inputs.cols() != targets.cols())
        throw std::invalid_argument("NeuralNetwork::BackpropagateBatch: inputs and targets must have the same number of columns");

    cost = 0.0;
    
    std::vector<Eigen::MatrixXd> weightGradients(m_layers.size());
    std::vector<Eigen::VectorXd> biasGradients(m_layers.size());
    
    for(int i = 0; i < inputs.cols(); i++){
        FeedForward(inputs.col(i));
        
        Layer& outputLayer = m_layers[m_layers.size() - 1];
        Eigen::VectorXd c = (outputLayer.GetActivations() - targets.col(i)); 
        Eigen::VectorXd outputError = Math::LeakyReLUDerivative(outputLayer.GetWeightedSums()).cwiseProduct(c);

        Eigen::MatrixXd outputWeightGradients = outputError * m_layers[m_layers.size() - 2].GetActivations().transpose();
        if(i == 0){
            weightGradients[m_layers.size() - 1] = outputWeightGradients;
            biasGradients[m_layers.size() - 1] = outputError;
        }
        else{
            weightGradients[m_layers.size() - 1] += outputWeightGradients;
            biasGradients[m_layers.size() - 1] += outputError;
        }
    

        for(int j = m_layers.size() - 2; j > 0; j--){
            Layer& hiddenLayer = m_layers[j];
            Layer& nextLayer = m_layers[j+1];

            Eigen::VectorXd hiddenError = Math::LeakyReLUDerivative(hiddenLayer.GetWeightedSums()).cwiseProduct(nextLayer.GetWeights().transpose() * outputError);

            Eigen::MatrixXd hiddenWeightGradients = hiddenError * m_layers[j-1].GetActivations().transpose();
            Eigen::VectorXd hiddenBiasGradients = hiddenError;

            if(i == 0){
                weightGradients[j] = hiddenWeightGradients;
                biasGradients[j] = hiddenBiasGradients;
            }
            else{
                weightGradients[j] += hiddenWeightGradients;
                biasGradients[j] += hiddenBiasGradients;
            }
            outputError = hiddenError;
        }
        cost += Math::MeanSquaredError(outputLayer.GetActivations(), targets.col(i));
    }
    for(int i = 1; i < weightGradients.size(); i++){
        weightGradients[i] /= inputs.cols();
        biasGradients[i] /= inputs.cols();

        m_layers[i].SetWeights(m_layers[i].GetWeights() - weightGradients[i] * learningRate);
        m_layers[i].SetBiases(m_layers[i].GetBiases() - biasGradients[i] * learningRate);
    }
    cost /= inputs.cols();
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

