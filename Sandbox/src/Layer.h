#pragma once

#include "Neuron.h"

#include "Eigen/Dense"

#include <vector>

class Layer {

    public:
        //numInputs: number of neurons in previous layer
        Layer(int numInputs, int size) : m_size(size), m_weights(size, numInputs){
            std::cout << size << std::endl;
            for(int i = 0; i < size; i++){
                m_neurons.push_back(Neuron(numInputs));
                m_weights.row(i) << m_neurons[i].GetWeightsRef().transpose();
            }


        }
        
    

    private:
        std::vector<Neuron> m_neurons;
        //Number of rows = number of neurons
        Eigen::MatrixXd m_weights;

        int m_size;

    

};
