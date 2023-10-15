#pragma once

#include "Model/Encoder.h"
#include "Model/Decoder.h"

#include <vector>

// https://browse.arxiv.org/pdf/1606.05908.pdf

//Describes the parameters of the latent distribution
class VAE
{
    public:
        VAE(std::vector<unsigned int> encoderLayers, std::vector<unsigned int> decoderLayers) : 
            m_encoder(encoderLayers), m_decoder(decoderLayers) {m_latentSize = encoderLayers.back();};

        void Train(TrainingSettings settings, const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& targets, bool verbose=false);
        
        Eigen::VectorXd GetPrediction(const Eigen::VectorXd& input);

        Eigen::VectorXd FeedForward(const Eigen::VectorXd& inputs);

        void Backpropagate(const Eigen::MatrixXd& inputs, TrainingSettings settings, 
            double& cost, double epoch);

        void BackpropagateBatch(const Eigen::VectorXd& inputs, TrainingSettings settings, 
            double& cost, double epoch);
    // private:
        Eigen::VectorXd CalculateLatent(const QParams& qParams);
        
    
    Encoder m_encoder;
    Decoder m_decoder;

    public:
        QParams m_qParams;
        int m_latentSize;

};

