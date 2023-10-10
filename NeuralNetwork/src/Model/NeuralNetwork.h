#pragma once

#include "Layer.h"

#include "Eigen/Dense"

#include <vector>
#include <iostream>



struct TrainingSettings{
    TrainingSettings(int epochs, int batchSize, double learningRate, double beta1=0.9, double beta2=0.999, double epsilon=1e-8) 
        : learningRate(learningRate), epochs(epochs), batchSize(batchSize), beta1(beta1), beta2(beta2), epsilon(epsilon) {}

    double learningRate;
    int epochs;
    int batchSize;

    //Adam parameters
    double beta1;
    double beta2;
    double epsilon;

};

class NeuralNetwork {
    public:
        NeuralNetwork(std::vector<int> topology); 

        void Backpropagate(const Eigen::VectorXd input, const Eigen::VectorXd& target, double learningRate, double& cost);

        void Train(TrainingSettings settings, const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& targets, bool verbose=false);

        Eigen::VectorXd GetPrediction(const Eigen::VectorXd& input) {
            FeedForward(input);
            return m_layers[m_layers.size() - 1].GetActivations();
        };

        ~NeuralNetwork() = default;
    private:
        struct MomentGradients{

        std::vector<Eigen::MatrixXd> m_w;
        std::vector<Eigen::VectorXd> m_b; 

        std::vector<Eigen::MatrixXd> v_w;
        std::vector<Eigen::VectorXd> v_b;

        };
        void FeedForward(const Eigen::VectorXd& input);
        void BackpropagateBatch(const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& targets, TrainingSettings settings, 
            double& cost, double epoch);

        MomentGradients m_momentGradients;
        std::vector<int> m_topology;
        std::vector<Layer> m_layers;


};

