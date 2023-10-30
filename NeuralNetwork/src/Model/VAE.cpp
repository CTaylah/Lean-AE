#include "VAE.h"


Eigen::VectorXd VAE::FeedForward(const Eigen::VectorXd& inputs){
    m_qParams = m_encoder.Encode(inputs);
    Eigen::VectorXd latent = CalculateLatent(m_qParams);
    return m_decoder.Decode(latent);
}

Eigen::VectorXd VAE::Decode(const Eigen::VectorXd& input){
    return m_decoder.Decode(input);
}

Eigen::VectorXd VAE::CalculateLatent(const QParams& qParams){
    return qParams.mu + (qParams.eps.array() * qParams.logVar.array().exp().sqrt()).matrix();
}

void VAE::Train(TrainingSettings settings, const Eigen::MatrixXd& inputs, bool verbose){
    double KLweight = 0;
    double MSEweight = 1;
    double totalCost = 0.0;
    int numBatches = inputs.cols() / settings.batchSize;
    for(int epoch = 0; epoch < settings.epochs; epoch++){
        double cost = 0.0;
        if(epoch > 10)
        {
            KLweight = 0.1;
            MSEweight = 1 - KLweight;
        }
        for(size_t i = 0; i < numBatches; i++)
        {
            Eigen::MatrixXd batch = inputs.block(0, i * settings.batchSize, inputs.rows(), settings.batchSize);
            Backpropagate(batch, settings, cost, epoch, KLweight, MSEweight);
        }

        totalCost += cost;
        if(verbose && epoch % 2 == 0)
            std::cout << "Epoch: " << epoch << " Cost: " << totalCost / epoch << std::endl;
    }
}

void VAE::Backpropagate(const Eigen::MatrixXd& inputs, TrainingSettings settings, 
    double& cost, int epoch, double KLweight, double MSEweight)
{
    std::vector<QParams> qParams(inputs.cols());
    //Create a set of latents for mini batch updates
    Eigen::MatrixXd latents = Eigen::MatrixXd::Zero(m_latentSize, inputs.cols());
    for(size_t i = 0; i < inputs.cols(); i++)
    {
        qParams[i] = m_encoder.Encode(inputs.col(i));
        latents.col(i) = CalculateLatent(qParams[i]);
    }
    
    std::vector<Eigen::VectorXd> decoderError = m_decoder.Backpropagate(latents, inputs, settings, epoch);
    m_encoder.Backpropagate(inputs, decoderError, qParams, settings, epoch, KLweight, MSEweight);

    //KL divergence cost.
    //This doesn't need to be calculated every backprop call
    Eigen::VectorXd mu_p = Eigen::VectorXd::Zero(m_latentSize);
    Eigen::VectorXd logvar_p = Eigen::VectorXd::Constant(m_latentSize, 1.0);

    // m_qParams = m_encoder.Encode(inputs);

    double miniBatchCost = 0;
    for(size_t sample; sample < inputs.cols(); sample++){
        Eigen::VectorXd latent = CalculateLatent(qParams[sample]);
        Eigen::VectorXd prediction = m_decoder.Decode(latent);

        double costMSE = Math::MeanSquaredError(prediction, inputs.col(sample));
        //Maybe should be flipped
        
        double costKL = Math::KL(mu_p, logvar_p, qParams[sample].mu, qParams[sample].logVar);

        miniBatchCost += MSEweight * costMSE + KLweight * costKL;
    }
    miniBatchCost /= inputs.cols();
    cost += miniBatchCost;
}

