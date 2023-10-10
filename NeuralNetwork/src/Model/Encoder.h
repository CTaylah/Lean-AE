#pragma once
#include "Model/Layer.h"
#include "NeuralNetwork.h"

struct QParams
{
    Eigen::VectorXd mu;
    Eigen::VectorXd logVar;
    Eigen::VectorXd eps;

    Eigen::VectorXd z;
};

class Encoder
{
    public:
        Encoder(std::vector<unsigned int> layers);
        void Backpropagate(Eigen::VectorXd inputs, Eigen::VectorXd target, Eigen::VectorXd decoderError, QParams qParams, 
            TrainingSettings settings, int epoch);
        QParams Encode(const Eigen::VectorXd& input);
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
