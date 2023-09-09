#pragma once

#include "Neuron.h"

#include "Eigen/Dense"

#include <vector>

class Layer {

    public:
        //numInputs: number of neurons in previous layer
        Layer(int numInputs, int size) : m_size(size){
            if(numInputs < 0 || size <= 0){
                throw std::invalid_argument("Layer: invalid number of inputs or size");
            }

            m_weights = Eigen::MatrixXd::Random(size, numInputs);
            m_biases = Eigen::VectorXd::Random(size);
        }
        
        Eigen::VectorXd FeedForward(const Eigen::VectorXd& inputs);

        Eigen::VectorXd GetActivations(){ return m_activations; }
        Eigen::VectorXd GetWeightedSums(){ return m_weightedSums; }
        Eigen::MatrixXd GetWeights(){ return m_weights; }
        Eigen::VectorXd GetBiases(){ return m_biases; }

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
        //Number of rows = number of neurons
        Eigen::MatrixXd m_weights;
        Eigen::VectorXd m_biases;
        Eigen::VectorXd m_activations;
        Eigen::VectorXd m_weightedSums;

        int m_size;
    

};

