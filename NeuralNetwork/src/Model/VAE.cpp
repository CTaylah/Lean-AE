#include "VAE.h"


Eigen::VectorXd VAE::FeedForward(const Eigen::VectorXd& inputs){
    m_qParams = m_encoder.Encode(inputs);
    m_qParams.eps = Math::genGaussianVector(m_qParams.mu.size(), 0, 1);
    m_qParams.z = m_qParams.mu + (m_qParams.eps.array() * m_qParams.logVar.array().exp().sqrt()).matrix();
    return m_decoder.Decode(m_qParams.z);
}


void VAE::Backpropagate(const Eigen::VectorXd& inputs, TrainingSettings settings, 
    double& cost, double epoch)
{
    Eigen::VectorXd prediction = FeedForward(inputs);
    Eigen::VectorXd decoderError = m_decoder.Backpropagate(m_qParams.z, inputs, settings, epoch);
    m_encoder.Backpropagate(inputs, inputs, decoderError, m_qParams, settings, epoch);

    double costMSE = Math::MeanSquaredError(prediction, inputs);
    Eigen::VectorXd mu_p = Eigen::VectorXd::Zero(m_qParams.mu.size());
    Eigen::VectorXd logvar_p = Eigen::VectorXd::Constant(m_qParams.mu.size(), 0.5);
    double costKL = Math::KL(mu_p, logvar_p, m_qParams.mu, m_qParams.logVar);

    cost = costMSE + costKL;
}