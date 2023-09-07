#include <vector>
#include <iostream>
#include <numeric>

#include "Eigen/Dense"


/*
    Structure consists of 3 main classes:
    Networks, which consist of Layers
    Layers, which consist of Neurons
    Neurons, which do math (sigmoid activation)

*/


//Sigmoid neuron 
class Neuron{
    public:

        Neuron(int inputCount) {
            m_weights = Eigen::VectorXd::Random(inputCount);
        }

        //Uses sigmoid function
        float getActivation(const Eigen::VectorXd& inputs) const{
            double weightedSum = 0;
            for (int i = 0; i < inputs.rows(); i++){
                weightedSum += inputs(i) * m_weights(i);
            } 
            weightedSum += m_bias;
            m_weightedSum = weightedSum;
            return 1.0 / (1 + std::exp(-(weightedSum)));
        }

        float getSigmoidDerivative() const{
            double sigZ = 1.0/ (1 + std::exp(-m_weightedSum));
            return sigZ * (1 - sigZ);
        }

    

    private:
        Eigen::VectorXd m_weights;
        //IDK ABOUT THIS BUT ERROR WENT AWAY
        mutable double m_weightedSum = 0;
        double m_bias = 0;

    public:
        const Eigen::VectorXd& getWeights(){
            return m_weights;
        }
        void setWeights(const Eigen::VectorXd weights)
        {
            m_weights = weights;
        }
        void setWeight(int index, double weight)
        {
            m_weights(index) = weight;
        }

        double getBias() { return m_bias; }
        void setBias(double bias) { m_bias = bias;}
      
      
};

class Layer{
    public:
        std::vector<Neuron> m_neurons;
        //(#inputs, #outputs)
        Layer(int inputCount, int neuronCount) {
            
            m_weightMatrix.resize(inputCount, neuronCount);

            for (int i = 0; i < neuronCount; i++){
                m_neurons.push_back(Neuron(inputCount));
                m_weightMatrix.col(i) = m_neurons[i].getWeights();
            }
        }

        //Take inputs from previous layer to update neurons
        Eigen::VectorXd getActivations(const Eigen::VectorXd& input) const{
            Eigen::VectorXd outputs(m_neurons.size());
            //Feed each neuron the previous layers' outputs
            for (int i = 0; i < m_neurons.size(); i++){
                outputs(i) = m_neurons[i].getActivation(input);
            }
            m_outputs = outputs;
            return outputs;
        }

        void updateWeightMatrix(){
            std::cout << m_weightMatrix.rows() << " " << m_weightMatrix.cols() << std::endl;
            for (int i = 0; i < m_neurons.size(); i++){
                std::cout << m_neurons[i].getWeights().rows() << std::endl;
                m_weightMatrix.row(i) = m_neurons[i].getWeights();
            }
        }

        Eigen::VectorXd getSigmoidDerivativeVector() const{
            Eigen::VectorXd sigmoidDerivativeVector(m_neurons.size());
            for (int i = 0; i < m_neurons.size(); i++){
                sigmoidDerivativeVector(i) = m_neurons[i].getSigmoidDerivative();
            }
            return sigmoidDerivativeVector;
        }
 

    private:
        //This should probably be a Map<Matrix>, but that's for another day
        Eigen::MatrixXd m_weightMatrix;
        //IDK ABOUT THIS BUT IT MADE SQUIGGLE GO AWAY
        mutable Eigen::VectorXd m_outputs;
        

    public:
        int getNeuronCount() const{
            return m_neurons.size();
        }
        
        Eigen::MatrixXd getWeightMatrix()
        {
            return m_weightMatrix;
        }

        Eigen::VectorXd getOutputs(){
            return m_outputs;
        }


        void printWeightMatrix(){
            updateWeightMatrix();
            std::cout << m_weightMatrix << std::endl;
        }

};

class Network{
    public:

        Network() {}

        Eigen::VectorXd feedForward(Eigen::VectorXd inputs){

            for(int i = 0; i < m_layers.size(); i++){
                inputs = m_layers[i].getActivations(inputs);
            }
            return inputs;
        };


        void push_back(Layer& layer)
        {
            m_layers.push_back(layer);
        }

        void learn(double learningRate, int epochs, int examples, const std::vector<Eigen::VectorXd>& labels, const std::vector<Eigen::VectorXd>& inputs)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double cost = 0;
                std::vector<int> indices(examples);
                std::iota(indices.begin(), indices.end(), 0);
                std::random_shuffle(indices.begin(), indices.end());

                for (int example = 0; example < examples; example++)
                {
                    Eigen::VectorXd layerActivations = feedForward(inputs[example]);
                    Eigen::VectorXd target = labels[example];

                    double meanSquaredError = (layerActivations - target).squaredNorm();
                    cost += meanSquaredError / 2;
                    // LayerError for Layer L
                    Eigen::VectorXd layerError = (layerActivations- target).cwiseProduct(m_layers.back().getSigmoidDerivativeVector());

                    for(int i = m_layers.size() - 1; i > 0; i--){
                        //For each neuron, calculate dC/dW
                        //std::cout << "Layer: " << i << std::endl;
                        for(int j = 0; j < m_layers[i].getNeuronCount(); j++)
                        {
                            //std::cout << "Neuron: " << j << std::endl;
                            Eigen::VectorXd previousActivations = m_layers[i-1].getOutputs();
                            for(int k = 0; k < previousActivations.rows(); k++){
                                double currentWeight = m_layers[i].m_neurons.at(j).getWeights()(k);
                                double currentBias = m_layers[i].m_neurons.at(j).getBias();

                                double dCdW = (learningRate) * (previousActivations(k) * layerError(j));
                                double dCdB = learningRate * layerError(j);

                                m_layers[i].m_neurons[j].setWeight(k,currentWeight - dCdW);
                                m_layers[i].m_neurons[j].setBias(currentBias - dCdB);
                                
                            }
                        }
                        //Update to LayerError for L - i
                        //m_layers[i].updateWeightMatrix();
                        layerError = (m_layers[i].getWeightMatrix() * layerError).cwiseProduct(m_layers[i-1].getSigmoidDerivativeVector());
                    }

                }
                cost = cost/examples;
                std::cout << "Epoch: " << epoch << std::endl;
                std::cout << "Cost: " << cost << std::endl;
                
            }
            
        }

    private: 
        std::vector<Layer> m_layers;

};











