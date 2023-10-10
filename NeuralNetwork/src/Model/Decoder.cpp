#include "Decoder.h"

Decoder::Decoder(std::vector<unsigned int> topology)
{
    m_layers.push_back(Layer(0, topology[0]));
    for(size_t i = 1; i < topology.size() - 1; i++){
        m_layers.push_back(Layer(topology[i-1], topology[i], ActivationFunction::LEAKY_RELU));
    }
    m_layers.push_back(Layer(topology[topology.size() - 2], topology.back(), ActivationFunction::SIGMOID));

    for(size_t i = 0; i < m_layers.size(); i++){
        m_momentGradients.m_w.push_back(Eigen::MatrixXd::Zero(m_layers[i].GetWeights().rows(), m_layers[i].GetWeights().cols()));
        m_momentGradients.m_b.push_back(Eigen::VectorXd::Zero(m_layers[i].GetBiases().rows()));

        m_momentGradients.v_w.push_back(Eigen::MatrixXd::Zero(m_layers[i].GetWeights().rows(), m_layers[i].GetWeights().cols()));
        m_momentGradients.v_b.push_back(Eigen::VectorXd::Zero(m_layers[i].GetBiases().rows()));
    }

}

Eigen::VectorXd Decoder::Decode(const Eigen::VectorXd& inputs) {
    Eigen::VectorXd l_inputs = inputs;

    for(size_t i = 0; i < m_layers.size(); i++){
        l_inputs = m_layers[i].FeedForward(l_inputs);
    }

    return m_layers.back().GetActivations();
}

Eigen::VectorXd Decoder::Backpropagate(Eigen::VectorXd latent, Eigen::VectorXd target, TrainingSettings settings, int epoch){
    std::vector<Eigen::MatrixXd> weightGradients;
    std::vector<Eigen::VectorXd> biasGradients;

    for(size_t i = 0; i < m_layers.size(); i++){
        weightGradients.push_back(Eigen::MatrixXd::Zero(m_layers[i].GetWeights().rows(), m_layers[i].GetWeights().cols()));
        biasGradients.push_back(Eigen::VectorXd::Zero(m_layers[i].GetBiases().rows()));
    }

    // Decode(latent);
    Eigen::VectorXd outputError = Eigen::VectorXd::Zero(m_layers.back().GetActivations().rows());
    for(size_t layer = m_layers.size() - 1; layer > 0; layer--)
    {
        if(layer == m_layers.size() - 1){
            Layer& outputLayer = m_layers[m_layers.size() - 1];
            Eigen::VectorXd c = (outputLayer.GetActivations() - target); 
            outputError = Math::LeakyReLUDerivative(outputLayer.GetWeightedSums()).cwiseProduct(c);
            Eigen::MatrixXd outputWeightGradients = outputError * m_layers[m_layers.size() - 2].GetActivations().transpose();

            weightGradients[layer] += outputWeightGradients;
            biasGradients[layer] += outputError;
            
            continue;
        }

        Layer& hiddenLayer = m_layers[layer];
        Layer& nextLayer = m_layers[layer+1];

        Eigen::VectorXd hiddenError = Math::LeakyReLUDerivative(hiddenLayer.GetWeightedSums()).cwiseProduct(nextLayer.GetWeights().transpose() * outputError);

        Eigen::MatrixXd hiddenWeightGradients = hiddenError * m_layers[layer-1].GetActivations().transpose();
        Eigen::VectorXd hiddenBiasGradients = hiddenError;

        weightGradients[layer] += hiddenWeightGradients;
        biasGradients[layer] += hiddenBiasGradients;

        outputError = hiddenError;

    }


    for(size_t layerIndex = 1; layerIndex < m_layers.size(); layerIndex++){
        weightGradients[layerIndex] /= latent.cols();
        biasGradients[layerIndex] /= latent.cols();

        //https://arxiv.org/abs/1412.6980

        // if(epoch == 0)
        // {
            m_layers[layerIndex].SetWeights(m_layers[layerIndex].GetWeights() - settings.learningRate * weightGradients[layerIndex]);
            m_layers[layerIndex].SetBiases(m_layers[layerIndex].GetBiases() - settings.learningRate * biasGradients[layerIndex]);
        //     continue;
        // }

        // int t = epoch;

        // m_momentGradients.m_w[layerIndex] = settings.beta1 * m_momentGradients.m_w[layerIndex] + (1 - settings.beta1) * weightGradients[layerIndex];
        // m_momentGradients.m_b[layerIndex] = settings.beta1 * m_momentGradients.m_b[layerIndex] + (1 - settings.beta1) * biasGradients[layerIndex];

        // m_momentGradients.v_w[layerIndex] = settings.beta2 * m_momentGradients.v_w[layerIndex] 
        //     + (1 - settings.beta2) * weightGradients[layerIndex].cwiseProduct(weightGradients[layerIndex]);

        // m_momentGradients.v_b[layerIndex] = settings.beta2 * m_momentGradients.v_b[layerIndex] 
        //     + (1 - settings.beta2) * biasGradients[layerIndex].cwiseProduct(biasGradients[layerIndex]);

        // Eigen::MatrixXd m_w_hat = m_momentGradients.m_w[layerIndex] / (1 - std::pow(settings.beta1, t));
        // Eigen::VectorXd m_b_hat = m_momentGradients.m_b[layerIndex] / (1 - std::pow(settings.beta1, t));

        // Eigen::MatrixXd v_w_hat = m_momentGradients.v_w[layerIndex] / (1 - std::pow(settings.beta2, t));
        // Eigen::VectorXd v_b_hat = m_momentGradients.v_b[layerIndex] / (1 - std::pow(settings.beta2, t));

        // Eigen::MatrixXd newWeights = m_layers[layerIndex].GetWeights() 
        //     - settings.learningRate * m_w_hat.cwiseQuotient((v_w_hat.cwiseSqrt().array() + settings.epsilon).matrix());  

        // Eigen::VectorXd newBiases = m_layers[layerIndex].GetBiases() 
        //     - settings.learningRate * m_b_hat.cwiseQuotient((v_b_hat.cwiseSqrt().array() + settings.epsilon).matrix());


        // m_layers[layerIndex].SetWeights(newWeights);
        // m_layers[layerIndex].SetBiases(newBiases);

    }

        outputError = m_layers[1].GetWeights().transpose() * outputError;

    return outputError;

}
