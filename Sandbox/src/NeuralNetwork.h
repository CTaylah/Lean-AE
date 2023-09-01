#include <vector>
#include <iostream>
#include "Eigen/Dense"

//Sigmoid neuron 
class Neuron{
    public:

        Neuron(int inputCount) {
            m_weights = Eigen::VectorXd::Random(inputCount);
        }

        //Uses sigmoid function
        inline float getOutput(const Eigen::VectorXd& inputs) const{
            double weightedSum = 0;
            for (int i = 0; i < inputs.rows(); i++){
                weightedSum += inputs(i) * m_weights(i);
            } 
            return 1.0 / (1 + std::exp(weightedSum - bias));
        }
    private:
        Eigen::VectorXd m_weights;
        float bias = 0;

      
};

class Layer{
    private:
        std::vector<Neuron> m_neurons;

    public:
        //Input count is the previous layers' neuron count
        Layer(int neuronCount, int inputCount) {
            for (int i = 0; i < neuronCount; i++){
                m_neurons.push_back(Neuron(inputCount));
            }
        }

        //Rename? Take inputs from previous layer to update neurons
        Eigen::VectorXd getOutputs(const Eigen::VectorXd& input) const{
            Eigen::VectorXd outputs(m_neurons.size());
            //Feed each neuron the previous layers' outputs
            for (int i = 0; i < m_neurons.size(); i++){
                outputs(i) = m_neurons[i].getOutput(input);
            }
            return outputs;
        }

    public:
        int getNeuronCount() const{
            return m_neurons.size();
        }

    
};

class Network{
    public:

        Network(const Eigen::VectorXd& inputs) : m_inputs(inputs){}

        Eigen::VectorXd feedForward(){
            auto inputs = m_inputs;
            for(int i = 0; i < m_hiddenLayers.size(); i++){
                inputs = m_hiddenLayers[i].getOutputs(inputs);
            }

            return inputs;
        };

        void push_back(Layer& layer)
        {
            m_hiddenLayers.push_back(layer);
        }

    private: 
        const Eigen::VectorXd& m_inputs;
        std::vector<Layer> m_hiddenLayers;
};

















