#include "Model/NeuralNetwork.h"
#include "Model/VAE.h"
#include "Common/Data.h"
#include "Testing/CrossValidation.h"

#include <iostream>
#include <ctime>
#include <omp.h>
#include <chrono>
int main(int argc, char** argv){

    srand(time(NULL));
    int num_threads = 1;
    omp_set_num_threads(num_threads);

    Eigen::MatrixXi examples(784, 60000);

    Eigen::Ref<Eigen::MatrixXi> examplesRef = examples; 
    
    bool success = ReadMNISTImages("MNIST/training_images/train-images.idx3-ubyte", examplesRef);

    if(!success){
        std::cout << "Failed to read MNIST data" << std::endl;
        return 1;
    }

    Eigen::MatrixXd examplesDouble = examplesRef.cast<double>();
    examplesDouble = examplesDouble / 255.0;

    //For testing purposes, only using subset of data
    Eigen::MatrixXd examplesSubset = examplesDouble.block(0, 0, examplesDouble.rows(), 100);

    // NeuralNetwork network1({784, 162, 784});
    TrainingSettings settings(30, 30, 0.00099);

    std::vector<unsigned int> encoderTopology = {784, 500, 256};
    std::vector<unsigned int> decoderTopology = {256, 500, 784};
    VAE vae(encoderTopology, decoderTopology);

    vae.Train(settings, examplesSubset, true);


    Eigen::VectorXd input = examplesSubset.col(0);
    Eigen::VectorXd prediction = vae.FeedForward(input);
    VectorToPPM((prediction * 255).cast<int>(), "output");
    VectorToPPM((input * 255).cast<int>(), "input");

    QParams qParams = vae.m_qParams;
    qParams.eps = Math::GenGaussianVector(qParams.mu.size(), 0, 1);
    Eigen::VectorXd latent = VAE::CalculateLatent(qParams);
    Eigen::VectorXd prediction2 = vae.Decode(latent);
    VectorToPPM((prediction2 * 255).cast<int>(), "output2");
}
