#include "NeuralNetwork.h"
#include <Eigen/StdVector>
#include <omp.h>

NeuralNetwork::NeuralNetwork(std::vector<int> topology) : m_topology(topology){
    if(m_topology.size() < 2)
        throw std::invalid_argument("Topology must have at least 2 layers");

    m_layers.push_back(Layer(0, m_topology[0]));

    for(int i = 1; i < m_topology.size(); i++){
        m_layers.push_back(Layer(m_topology[i-1], m_topology[i]));
    }
}

void NeuralNetwork::FeedForward(const Eigen::VectorXd& inputs){
    Eigen::VectorXd l_inputs = inputs;

    for(int i = 0; i < m_layers.size(); i++){
        l_inputs = m_layers[i].FeedForward(l_inputs);
    }
}

void NeuralNetwork::Train(TrainingSettings settings, const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& targets){

    if(inputs.cols() != targets.cols())
        throw std::invalid_argument("NeuralNetwork::Train: inputs and targets must have the same number of columns");

        for(int i = 0; i < m_layers.size(); i++){
            settings.firstMomentWeightGradients.push_back(Eigen::MatrixXd::Zero(m_layers[i].GetWeights().rows(), m_layers[i].GetWeights().cols()));
            settings.firstMomentBiasGradients.push_back(Eigen::VectorXd::Zero(m_layers[i].GetBiases().rows()));

            settings.secondMomentWeightGradients.push_back(Eigen::MatrixXd::Zero(m_layers[i].GetWeights().rows(), m_layers[i].GetWeights().cols()));
            settings.secondMomentBiasGradients.push_back(Eigen::VectorXd::Zero(m_layers[i].GetBiases().rows()));
        }


    double numBatches = inputs.cols() / settings.batchSize;
    double totalCost = 0.0;
    for(int epoch = 0; epoch < settings.epochs; epoch++){
        double cost = 0.0;
        for(int i = 0; i < numBatches; i++){
            Eigen::MatrixXd batch = inputs.block(0, i * settings.batchSize, inputs.rows(), settings.batchSize);
            Eigen::MatrixXd batchTargets = targets.block(0, i, targets.rows(), settings.batchSize);
            BackpropagateBatch(batch, batchTargets, settings, cost, epoch);
            
        }
        cost /= numBatches;
        totalCost += cost;
        std::cout << "Epoch: " << epoch << " Cost: " << totalCost / epoch << std::endl;
    }
}


void NeuralNetwork::BackpropagateBatch(const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& targets, TrainingSettings settings, double& cost, double epoch){
    if(inputs.cols() != targets.cols())
        throw std::invalid_argument("NeuralNetwork::BackpropagateBatch: inputs and targets must have the same number of columns");

    double miniBatchCost = 0.0;
    
    std::vector<Eigen::MatrixXd> weightGradients(m_layers.size());
    std::vector<Eigen::VectorXd> biasGradients(m_layers.size());


    //    #pragma omp parallel
    {
        std::vector<Eigen::MatrixXd> privateWeightGradients(m_layers.size());
        std::vector<Eigen::VectorXd> privateBiasGradients(m_layers.size());

        // #pragma omp for
    for(int i = 0; i < inputs.cols(); i++){
        FeedForward(inputs.col(i));
        
        Layer& outputLayer = m_layers[m_layers.size() - 1];
        Eigen::VectorXd c = (outputLayer.GetActivations() - targets.col(i)); 
        Eigen::VectorXd outputError = Math::LeakyReLUDerivative(outputLayer.GetWeightedSums()).cwiseProduct(c);

        Eigen::MatrixXd outputWeightGradients = outputError * m_layers[m_layers.size() - 2].GetActivations().transpose();
        //This might need to be single threaded
        if(i == 0){
            privateWeightGradients[m_layers.size() - 1] = outputWeightGradients;
            privateBiasGradients[m_layers.size() - 1] = outputError;
        }
        
        else{
            privateWeightGradients[m_layers.size() - 1] += outputWeightGradients;
            privateBiasGradients[m_layers.size() - 1] += outputError;
        }
    
        //Calculate gradient vectors and matrices for hdden layers
        for(int j = m_layers.size() - 2; j > 0; j--){
            Layer& hiddenLayer = m_layers[j];
            Layer& nextLayer = m_layers[j+1];

            Eigen::VectorXd hiddenError = Math::LeakyReLUDerivative(hiddenLayer.GetWeightedSums()).cwiseProduct(nextLayer.GetWeights().transpose() * outputError);

            Eigen::MatrixXd hiddenWeightGradients = hiddenError * m_layers[j-1].GetActivations().transpose();
            Eigen::VectorXd hiddenBiasGradients = hiddenError;

            if(i == 0){
                privateWeightGradients[j] = hiddenWeightGradients;
                privateBiasGradients[j] = hiddenBiasGradients;
            }
            else{
                privateWeightGradients[j] += hiddenWeightGradients;
                privateBiasGradients[j] += hiddenBiasGradients;
            }
            outputError = hiddenError;
        }
        miniBatchCost += Math::MeanSquaredError(outputLayer.GetActivations(), targets.col(i));
    }
    miniBatchCost /= inputs.cols();

    cost += miniBatchCost;

    // #pragma omp critical
    {
        for(int layerIndex = 1; layerIndex < m_layers.size(); layerIndex++){
                weightGradients[layerIndex] = privateWeightGradients[layerIndex];
                biasGradients[layerIndex] = privateBiasGradients[layerIndex];
            }
        }
    }


    for(int layerIndex = 1; layerIndex < weightGradients.size(); layerIndex++){
        weightGradients[layerIndex] /= inputs.cols();
        biasGradients[layerIndex] /= inputs.cols();

        //Github copilot garbledy goop that supposedly makes Adam work
        //https://arxiv.org/abs/1412.6980

        if(epoch == 0)
        {
            m_layers[layerIndex].SetWeights(m_layers[layerIndex].GetWeights() - settings.learningRate * weightGradients[layerIndex]);
            m_layers[layerIndex].SetBiases(m_layers[layerIndex].GetBiases() - settings.learningRate * biasGradients[layerIndex]);
            continue;
        }
        int t = epoch; //+ 1;

        settings.firstMomentWeightGradients[layerIndex] = settings.beta1 * settings.firstMomentWeightGradients[layerIndex] + (1 - settings.beta1) * weightGradients[layerIndex];
        settings.firstMomentBiasGradients[layerIndex] = settings.beta1 * settings.firstMomentBiasGradients[layerIndex] + (1 - settings.beta1) * biasGradients[layerIndex];

        settings.secondMomentWeightGradients[layerIndex] = settings.beta2 * settings.secondMomentWeightGradients[layerIndex] + (1 - settings.beta2) * weightGradients[layerIndex].cwiseProduct(weightGradients[layerIndex]);
        settings.secondMomentBiasGradients[layerIndex] = settings.beta2 * settings.secondMomentBiasGradients[layerIndex] + (1 - settings.beta2) * biasGradients[layerIndex].cwiseProduct(biasGradients[layerIndex]);

        Eigen::MatrixXd firstMomentWeightGradientsCorrected = settings.firstMomentWeightGradients[layerIndex] / (1 - std::pow(settings.beta1, t));
        Eigen::VectorXd firstMomentBiasGradientsCorrected = settings.firstMomentBiasGradients[layerIndex] / (1 - std::pow(settings.beta1, t));

        Eigen::MatrixXd secondMomentWeightGradientsCorrected = settings.secondMomentWeightGradients[layerIndex] / (1 - std::pow(settings.beta2, t));
        Eigen::VectorXd secondMomentBiasGradientsCorrected = settings.secondMomentBiasGradients[layerIndex] / (1 - std::pow(settings.beta2, t));

        Eigen::MatrixXd newWeights = m_layers[layerIndex].GetWeights() - settings.learningRate * firstMomentWeightGradientsCorrected.cwiseQuotient((secondMomentWeightGradientsCorrected.cwiseSqrt().array() + settings.epsilon).matrix());  
        Eigen::VectorXd newBiases = m_layers[layerIndex].GetBiases() - settings.learningRate * firstMomentBiasGradientsCorrected.cwiseQuotient((secondMomentBiasGradientsCorrected.cwiseSqrt().array() + settings.epsilon).matrix());


        m_layers[layerIndex].SetWeights(newWeights);
        m_layers[layerIndex].SetBiases(newBiases);

    }
}
