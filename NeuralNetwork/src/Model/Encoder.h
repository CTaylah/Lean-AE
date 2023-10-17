#pragma once
#include "Model/Layer.h"
#include "NeuralNetwork.h"

struct QParams
{
    Eigen::VectorXd mu;
    Eigen::VectorXd logVar;
    Eigen::VectorXd eps;
};

class Encoder
{
    public:
        Encoder(std::vector<unsigned int> layers);

        void Backpropagate(const Eigen::MatrixXd& inputs, const Eigen::MatrixXd& target, std::vector<Eigen::VectorXd> decoderError, std::vector<QParams> qParams, 
            TrainingSettings settings, int epoch, double klWeight);

        QParams Encode(const Eigen::MatrixXd& input);
    private:
        std::vector<Layer> m_layers;
        
        //Adam update
        void UpdateParameters(const std::vector<Eigen::MatrixXd>& weightGradients, const std::vector<Eigen::VectorXd>& biasGradients, 
            const TrainingSettings& settings, int epoch);

        struct MomentGradients{
        std::vector<Eigen::MatrixXd> m_w;
        std::vector<Eigen::VectorXd> m_b; 

        std::vector<Eigen::MatrixXd> v_w;
        std::vector<Eigen::VectorXd> v_b;
        };

        MomentGradients m_momentGradients;
};
