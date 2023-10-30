#include "Encoder.h"

Encoder::Encoder(std::vector<unsigned int> topology)
{
    m_layers.push_back(Layer(0, topology[0]));
    for(size_t i = 1; i < topology.size() - 1; i++){
        m_layers.push_back(Layer(topology[i-1], topology[i], ActivationFunction::LEAKY_RELU));
    }
    m_layers.push_back(Layer(topology[topology.size() - 2], topology.back(), ActivationFunction::SOFTPLUS));

    //Moment gradients are initialized to 0
    for(size_t i = 0; i < m_layers.size(); i++){
        m_momentGradients.m_w.push_back(Eigen::MatrixXd::Zero(m_layers[i].GetWeights().rows(), m_layers[i].GetWeights().cols()));
        m_momentGradients.m_b.push_back(Eigen::VectorXd::Zero(m_layers[i].GetBiases().rows()));

        m_momentGradients.v_w.push_back(Eigen::MatrixXd::Zero(m_layers[i].GetWeights().rows(), m_layers[i].GetWeights().cols()));
        m_momentGradients.v_b.push_back(Eigen::VectorXd::Zero(m_layers[i].GetBiases().rows()));
    }
}

QParams Encoder::Encode(const Eigen::VectorXd& inputs){
    Eigen::VectorXd l_inputs = inputs;

    for(size_t i = 0; i < m_layers.size(); i++){
        l_inputs = m_layers[i].FeedForward(l_inputs);
    }

    Eigen::VectorXd logvar = l_inputs; 
    //Dubious but chatGPT said it was ok
    Eigen::VectorXd mu = m_layers.back().GetWeightedSums();
    Eigen::VectorXd eps = Math::GenGaussianVector(mu.size(), 0, 1);
    return QParams{mu, logvar, eps};
}

// QParams Encoder::Encode(const Eigen::MatrixXd& inputs){
//    sQParams qParams;
//     for(size_t i = 0; i < inputs.cols(); i++){
//         qParams.push_back(Encode(inputs.col(i)));
//     }
//     return qParams;
// }

void Encoder::Backpropagate(const Eigen::MatrixXd& inputs, std::vector<Eigen::VectorXd> decoderError, std::vector<QParams> qParams, 
    TrainingSettings settings, int epoch, double KLweight, double MSEweight) //Not sure I want klWeight here
{

    //Two sets of gradients, one for reconstruction error, one for KL divergence
    //Every index in the vector is the gradient for that layer
    std::vector<Eigen::MatrixXd> MSEweightGradients;
    std::vector<Eigen::VectorXd> MSEbiasGradients;

    std::vector<Eigen::MatrixXd> KLweightGradients;
    std::vector<Eigen::VectorXd> KLbiasGradients;

    for(size_t i = 0; i < m_layers.size(); i++){
        MSEweightGradients.push_back(Eigen::MatrixXd::Zero(m_layers[i].GetWeights().rows(), m_layers[i].GetWeights().cols()));
        MSEbiasGradients.push_back(Eigen::VectorXd::Zero(m_layers[i].GetBiases().rows()));

        KLweightGradients.push_back(Eigen::MatrixXd::Zero(m_layers[i].GetWeights().rows(), m_layers[i].GetWeights().cols()));
        KLbiasGradients.push_back(Eigen::VectorXd::Zero(m_layers[i].GetBiases().rows()));
    }


for(size_t i = 0; i < inputs.cols(); i++){
    #pragma omp critical
    {
        Encode(inputs.col(i));
    }

    Eigen::VectorXd outputErrorRC;
    Eigen::VectorXd outputErrorDL;

   for(size_t layer = m_layers.size() - 1; layer > 0; layer--)
    {
        //Output layer calculations are different than the rest of the layers
        if(layer == m_layers.size() - 1){
            //Math here is dubious, probably incorrect
            Eigen::VectorXd variance = Math::UndoLog(qParams[i].logVar);
            Eigen::VectorXd halfLogVariance = qParams[i].logVar * 0.5; 
            Eigen::VectorXd e_halfLog = halfLogVariance.array().exp().matrix();
            Layer& outputLayer = m_layers[m_layers.size() - 1];
            Eigen::VectorXd del_dkl = Math::SoftplusDerivative(outputLayer.GetWeightedSums());
            Eigen::VectorXd dz_dlv = 0.5 * qParams[i].eps.cwiseProduct(variance.cwiseInverse()).cwiseProduct(qParams[i].mu).cwiseProduct(e_halfLog);
            outputErrorRC = del_dkl.cwiseProduct(dz_dlv).cwiseProduct(decoderError[i]); 
            outputErrorRC = del_dkl.cwiseProduct(outputErrorRC);
            

            Eigen::MatrixXd outputWeightGradientsRC = outputErrorRC * m_layers[m_layers.size() - 2].GetActivations().transpose();
                                                                                                        //What the hell is this? idk
            Eigen::VectorXd dDL_dV = 0.5 * (variance.array().inverse() - ((qParams[i].mu.array().square() + 0.0).cwiseProduct(variance.array().inverse().square())));
            Eigen::VectorXd dV_dLV = variance;
            Eigen::VectorXd dDL_dLV = dDL_dV.cwiseProduct(dV_dLV);

            outputErrorDL = dDL_dLV.cwiseProduct(del_dkl);
            Eigen::MatrixXd outputWeightGradientsDL = outputErrorDL * m_layers[m_layers.size() - 2].GetActivations().transpose();

            MSEweightGradients[layer] += outputWeightGradientsRC;
            MSEbiasGradients[layer] += outputErrorRC;

            KLweightGradients[layer] += outputWeightGradientsDL;
            KLbiasGradients[layer] += outputErrorDL;
            
            continue;
        }

        Layer& hiddenLayer = m_layers[layer];
        Layer& nextLayer = m_layers[layer+1];

        Eigen::VectorXd hiddenErrorRC = Math::LeakyReLUDerivative(hiddenLayer.GetWeightedSums()).cwiseProduct(nextLayer.GetWeights().transpose() * outputErrorRC);

        Eigen::MatrixXd hiddenWeightGradientsRC = hiddenErrorRC * m_layers[layer-1].GetActivations().transpose();
        Eigen::VectorXd hiddenBiasGradientsRC = hiddenErrorRC;
        
        Eigen::VectorXd hiddenErrorDL = Math::LeakyReLUDerivative(hiddenLayer.GetWeightedSums()).cwiseProduct(nextLayer.GetWeights().transpose() * outputErrorDL);

        Eigen::MatrixXd hiddenWeightGradientsDL = hiddenErrorDL * m_layers[layer-1].GetActivations().transpose();
        Eigen::VectorXd hiddenbiasGradientsDL = hiddenErrorDL; 

        MSEweightGradients[layer] += hiddenWeightGradientsRC;
        MSEbiasGradients[layer] += hiddenBiasGradientsRC;

        KLweightGradients[layer] += hiddenWeightGradientsDL;
        KLbiasGradients[layer] += hiddenbiasGradientsDL;

        outputErrorRC = hiddenErrorRC;
        outputErrorDL = hiddenErrorDL;
    } 
}
    std::vector<Eigen::MatrixXd> weightGradients;
    std::vector<Eigen::VectorXd> biasGradients;

    for(size_t i = 0; i < MSEweightGradients.size(); i++)
    {
        weightGradients.push_back(KLweight * KLweightGradients[i] + MSEweight * MSEweightGradients[i]);
        biasGradients.push_back(KLweight * KLbiasGradients[i] + MSEweight * MSEbiasGradients[i]);
    }

    UpdateParameters(weightGradients, biasGradients, settings, epoch);

}


void Encoder::UpdateParameters(const std::vector<Eigen::MatrixXd>& weightGradients, const std::vector<Eigen::VectorXd>& biasGradients, const TrainingSettings& settings, int epoch){
    #pragma omp parallel for
    for(size_t layerIndex = 1; layerIndex < m_layers.size(); layerIndex++){
        if(epoch == 0)
        {
            m_layers[layerIndex].SetWeights(m_layers[layerIndex].GetWeights() - settings.learningRate * weightGradients[layerIndex]);
            m_layers[layerIndex].SetBiases(m_layers[layerIndex].GetBiases() - settings.learningRate * biasGradients[layerIndex]);
        }
       //https://arxiv.org/abs/1412.6980

        if(epoch == 0)
        {
            m_layers[layerIndex].SetWeights(m_layers[layerIndex].GetWeights() - settings.learningRate * weightGradients[layerIndex]);
            m_layers[layerIndex].SetBiases(m_layers[layerIndex].GetBiases() - settings.learningRate * biasGradients[layerIndex]);
            continue;
        }

        int t = epoch;

        m_momentGradients.m_w[layerIndex] = settings.beta1 * m_momentGradients.m_w[layerIndex] + (1 - settings.beta1) * weightGradients[layerIndex];
        m_momentGradients.m_b[layerIndex] = settings.beta1 * m_momentGradients.m_b[layerIndex] + (1 - settings.beta1) * biasGradients[layerIndex];

        m_momentGradients.v_w[layerIndex] = settings.beta2 * m_momentGradients.v_w[layerIndex] 
            + (1 - settings.beta2) * weightGradients[layerIndex].cwiseProduct(weightGradients[layerIndex]);

        m_momentGradients.v_b[layerIndex] = settings.beta2 * m_momentGradients.v_b[layerIndex] 
            + (1 - settings.beta2) * biasGradients[layerIndex].cwiseProduct(biasGradients[layerIndex]);

        Eigen::MatrixXd m_w_hat = m_momentGradients.m_w[layerIndex] / (1 - std::pow(settings.beta1, t));
        Eigen::VectorXd m_b_hat = m_momentGradients.m_b[layerIndex] / (1 - std::pow(settings.beta1, t));

        Eigen::MatrixXd v_w_hat = m_momentGradients.v_w[layerIndex] / (1 - std::pow(settings.beta2, t));
        Eigen::VectorXd v_b_hat = m_momentGradients.v_b[layerIndex] / (1 - std::pow(settings.beta2, t));

        Eigen::MatrixXd newWeights = m_layers[layerIndex].GetWeights() 
            - settings.learningRate * m_w_hat.cwiseQuotient((v_w_hat.cwiseSqrt().array() + settings.epsilon).matrix());  

        Eigen::VectorXd newBiases = m_layers[layerIndex].GetBiases() 
            - settings.learningRate * m_b_hat.cwiseQuotient((v_b_hat.cwiseSqrt().array() + settings.epsilon).matrix());


        m_layers[layerIndex].SetWeights(newWeights);
        m_layers[layerIndex].SetBiases(newBiases);

    }
}