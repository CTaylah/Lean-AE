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

    private:
        std::vector<Neuron> m_neurons;
        //Number of rows = number of neurons
        Eigen::MatrixXd m_weights;
        Eigen::VectorXd m_biases;
        Eigen::VectorXd m_activations;

        int m_size;

    

};
