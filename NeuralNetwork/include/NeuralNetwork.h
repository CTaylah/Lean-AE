#pragma once

#include "Layer.h"

#include "Eigen/Dense"

#include <vector>
#include <iostream>

struct TrainingSettings{
    TrainingSettings(int epochs, int batchSize, double learningRate, double beta1=0.91, double beta2=0.999, double epsilon=1e-8) 
        : learningRate(learningRate), epochs(epochs), batchSize(batchSize), beta1(beta1), beta2(beta2), epsilon(epsilon) {}

    double learningRate;
    int epochs;
    int batchSize;

    //Adam parameters
    double beta1;
    double beta2;
    double epsilon;


    std::vector<Eigen::MatrixXd> firstMomentWeightGradients;
    std::vector<Eigen::VectorXd> firstMomentBiasGradients; 

    std::vector<Eigen::MatrixXd> secondMomentWeightGradients;
    std::vector<Eigen::VectorXd> secondMomentBiasGradients;

};

class NeuralNetwork {
    public:
        NeuralNetwork(std::vector<int> topology); 

        void Backpropagate(const Eigen::VectorXd input, const Eigen::VectorXd& target, double learningRate, double& cost);

        void Train(TrainingSettings settings, const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& targets);


        Eigen::VectorXd GetPrediction(const Eigen::VectorXd& input) {
            FeedForward(input);
            return m_layers[m_layers.size() - 1].GetActivations();
        };

        ~NeuralNetwork() = default;
    private:
        void FeedForward(const Eigen::VectorXd& input);
        void BackpropagateBatch(const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& targets, TrainingSettings settings, double& cost, double epoch);

        std::vector<int> m_topology;
        std::vector<Layer> m_layers;


};