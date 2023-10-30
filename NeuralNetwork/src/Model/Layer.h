#pragma once

#include "Common/Math.h"

#include "Eigen/Dense"

#include <vector>
#include <iostream>

enum class ActivationFunction{
    SIGMOID,
    RELU,
    LEAKY_RELU,
    SOFTPLUS

};

class Layer {

    public:
        //numInputs: number of neurons in previous layer
        Layer(int numInputs, int size, ActivationFunction actFunc=ActivationFunction::LEAKY_RELU) 
            : m_size(size), m_activationFunction(actFunc){

            if(numInputs < 0 || size <= 0){
                std::cout << numInputs << std::endl;
                std::cout << size << std::endl;
                std::cout << std::endl;
                throw std::invalid_argument("Layer: invalid number of inputs or size");
            } 

            m_weights = Eigen::MatrixXd::Random(size, numInputs) * sqrt(2.0 / numInputs);
            m_biases = Eigen::VectorXd::Zero(size);
        }
        
        Eigen::VectorXd FeedForward(const Eigen::VectorXd& inputs);

        Eigen::VectorXd GetActivations(){ return m_activations; }
        Eigen::VectorXd GetWeightedSums(){ return m_weightedSums; }
        Eigen::MatrixXd GetWeights(){ return m_weights; }
        Eigen::VectorXd GetBiases(){ return m_biases; }
        Eigen::VectorXd ActivationDerivative(const Eigen::VectorXd& activations);

        void SetWeights(const Eigen::MatrixXd& weights){ 
            if(weights.rows() != m_weights.rows() || weights.cols() != m_weights.cols())
                throw std::invalid_argument("Layer::SetWeights: invalid size of weights");

            m_weights = weights; 
        }

        void SetBiases(const Eigen::VectorXd& biases){ 
            if(biases.size() != m_biases.size())
                throw std::invalid_argument("Layer::SetBiases: invalid size of biases");

            m_biases = biases; 
        }

    private:
        ActivationFunction m_activationFunction;
        //Number of rows = number of neurons
        Eigen::MatrixXd m_weights;
        Eigen::VectorXd m_biases;
        Eigen::VectorXd m_weightedSums;

        Eigen::VectorXd m_activations;
        int m_size;

        Eigen::VectorXd CalculateActivation(const Eigen::VectorXd& weightedSums);

};

