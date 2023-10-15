#pragma once
#include "Model/Layer.h"
#include "NeuralNetwork.h"

class Decoder
{
    public:
        Decoder(std::vector<unsigned int> layers);
        //Laten variable Z
        Eigen::VectorXd Decode(const Eigen::VectorXd& latent);
        std::vector<Eigen::VectorXd> Backpropagate(const Eigen::MatrixXd& latent, const Eigen::MatrixXd& target, 
            TrainingSettings settings, int epoch);
    private:
        std::vector<Layer> m_layers;

        struct MomentGradients{
        std::vector<Eigen::MatrixXd> m_w;
        std::vector<Eigen::VectorXd> m_b; 

        std::vector<Eigen::MatrixXd> v_w;
        std::vector<Eigen::VectorXd> v_b;
        };

        MomentGradients m_momentGradients;
};
