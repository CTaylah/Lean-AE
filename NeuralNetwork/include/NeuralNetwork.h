#pragma once

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

struct AdamParameters{
    AdamParameters(double learningRate, double beta1, double beta2, double epsilon) 
        : learningRate(learningRate), beta1(beta1), beta2(beta2), epsilon(epsilon) {
            firstMomentWeightGradients = std::vector<Eigen::MatrixXd>(0);
            firstMomentBiasGradients = std::vector<Eigen::VectorXd>(0);

            secondMomentWeightGradients = std::vector<Eigen::MatrixXd>(0);
            secondMomentBiasGradients = std::vector<Eigen::VectorXd>(0);

            previousWeightGradients = std::vector<Eigen::MatrixXd>(0);
        }

    double learningRate;
    double beta1;
    double beta2;
    double epsilon;

    std::vector<Eigen::MatrixXd> firstMomentWeightGradients;
    std::vector<Eigen::VectorXd> firstMomentBiasGradients; 

    std::vector<Eigen::MatrixXd> secondMomentWeightGradients;
    std::vector<Eigen::VectorXd> secondMomentBiasGradients;

    std::vector<Eigen::MatrixXd> previousWeightGradients;

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

        AdamParameters m_adamParameters;

        struct AdamParameters;
};