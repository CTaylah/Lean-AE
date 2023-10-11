#include "VAE.h"


Eigen::VectorXd VAE::FeedForward(const Eigen::VectorXd& inputs){
    m_qParams = m_encoder.Encode(inputs);
    m_qParams.z = CalculateLatent(m_qParams.mu, m_qParams.logVar);
    return m_decoder.Decode(m_qParams.z);
}

Eigen::VectorXd VAE::CalculateLatent(const Eigen::VectorXd& means, const Eigen::VectorXd& logVariances){
    Eigen::VectorXd eps = Math::genGaussianVector(means.size(), 0, 1);
    return means + (eps.array() * logVariances.array().exp().sqrt()).matrix();
}

void VAE::Backpropagate(const Eigen::MatrixXd& inputs, TrainingSettings settings, 
    double& cost, double epoch)
{
    Eigen::MatrixXd latent = Eigen::MatrixXd::Zero(m_qParams.mu.size(), inputs.cols());
    for(size_t i = 0; i < inputs.cols(); i++)
    {

    }
    Eigen::VectorXd prediction = FeedForward(inputs);
    Eigen::VectorXd decoderError = m_decoder.Backpropagate(m_qParams.z, inputs, settings, epoch);
    m_encoder.Backpropagate(inputs, inputs, decoderError, m_qParams, settings, epoch);


    double costMSE = Math::MeanSquaredError(prediction, inputs);
    Eigen::VectorXd mu_p = Eigen::VectorXd::Zero(m_qParams.mu.size());
    Eigen::VectorXd logvar_p = Eigen::VectorXd::Constant(m_qParams.mu.size(), 0.5);
    double costKL = Math::KL(mu_p, logvar_p, m_qParams.mu, m_qParams.logVar);

    cost = costMSE + 0 * costKL;
}

void VAE::BackpropagateBatch(const Eigen::VectorXd& inputs, TrainingSettings settings,
    double& cost, double epoch)
{


}