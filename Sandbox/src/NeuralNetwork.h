#pragma once

#include "Layer.h"

#include "Eigen/Dense"

#include <vector>


struct Topology
{
    //Do not include input layer in neuronsPerLayer
    Topology(int inputSize, std::vector<unsigned int> neuronsPerLayer) 
        : inputSize(inputSize), neuronsPerLayer(neuronsPerLayer) {}
    unsigned int inputSize, outputSize;

    std::vector<unsigned int> neuronsPerLayer;

};

class NeuralNetwork {

    public:
        NeuralNetwork(Topology topology); 

        void FeedForward(const Eigen::VectorXd& input);
        
        ~NeuralNetwork() = default;

        Eigen::VectorXd GetPrediction(const Eigen::VectorXd& input);

    private:
        Topology m_topology;
        std::vector<Layer> m_layers;
};