#pragma once

#include "Data.h"
#include "Layer.h"

#include "Eigen/Dense"

#include <vector>
#include <iostream>

struct TrainingSettings{
    TrainingSettings(double learningRate, int epochs, int batchSize) 
        : learningRate(learningRate), epochs(epochs), batchSize(batchSize) {}

    double learningRate;
    int epochs;
    int batchSize;
};

class NeuralNetwork {

    public:
        NeuralNetwork(std::vector<int> topology); 

        void Backpropagate(const Eigen::VectorXd input, const Eigen::VectorXd& target, double learningRate, double& cost);

        void Train(TrainingSettings settings, const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& targets);

        void BackpropagateBatch(const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& targets, double learningRate, double& cost);

        Eigen::VectorXd GetPrediction(const Eigen::VectorXd& input) {
            FeedForward(input);
            return m_layers[m_layers.size() - 1].GetActivations();
        };

        ~NeuralNetwork() = default;
    private:
        void FeedForward(const Eigen::VectorXd& input);
        std::vector<int> m_topology;
        std::vector<Layer> m_layers;
};