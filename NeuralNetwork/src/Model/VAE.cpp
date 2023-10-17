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
    double klWeight = 0;
    int numBatches = inputs.cols() / settings.batchSize;
    for(int epoch = 0; epoch < settings.epochs; epoch++){
        for(size_t i = 0; i < numBatches; i++)
        {
            Eigen::MatrixXd batch = inputs.block(0, i * settings.batchSize, inputs.rows(), settings.batchSize);
            
        }
    }
}
void VAE::Backpropagate(const Eigen::MatrixXd& inputs, TrainingSettings settings, 
    double& cost, double epoch, double klWeight)
{
    std::vector<QParams> qParams(inputs.cols());
    Eigen::MatrixXd latents = Eigen::MatrixXd::Zero(m_latentSize, inputs.cols());
    for(size_t i = 0; i < inputs.cols(); i++)
    {
        qParams[i] = m_encoder.Encode(inputs.col(i));
        latents.col(i) = CalculateLatent(qParams[i]);
    }
    
    std::vector<Eigen::VectorXd> decoderError = m_decoder.Backpropagate(latents, inputs, settings, epoch);
    m_encoder.Backpropagate(inputs, inputs, decoderError, qParams, settings, epoch, klWeight);


    Eigen::VectorXd prediction = FeedForward(inputs);
    double costMSE = Math::MeanSquaredError(prediction, inputs);
    Eigen::VectorXd mu_p = Eigen::VectorXd::Zero(m_qParams.mu.size());
    Eigen::VectorXd logvar_p = Eigen::VectorXd::Constant(m_qParams.mu.size(), 0.5);
    double costKL = Math::KL(mu_p, logvar_p, m_qParams.mu, m_qParams.logVar);

    m_qParams = m_encoder.Encode(inputs);
    cost = costMSE + 0 * costKL;
}

void VAE::BackpropagateBatch(const Eigen::VectorXd& inputs, TrainingSettings settings,
    double& cost, double epoch)
{


}