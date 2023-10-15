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
    int num_threads = 2;
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
    Eigen::MatrixXd examplesSubset = examplesDouble.block(0, 0, examplesDouble.rows(), 10000);

    NeuralNetwork network1({784, 162, 784});
    TrainingSettings settings(10, num_threads * 12, 0.00087);
    NeuralNetwork network2({784, 1, 784});


    std::vector<NeuralNetwork> networks = {network1, network2};

    // auto start = std::chrono::high_resolution_clock::now();

    // // double cost = Testing::MonteCarloCV(network1, settings, examplesSubset, 0.7, true);

    // auto stop = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);
    // std::cout << duration.count() << std::endl;

    // std::cout << "Cost: " << cost << std::endl;
    //MonteCarloCV(network2, 0.8, examplesSubset, settings);

    std::vector<unsigned int> encoderTopology = {784, 500, 256};
    std::vector<unsigned int> decoderTopology = {256, 500, 784};
    VAE vae(encoderTopology, decoderTopology);
    double cost;
    for(int epoch = 0; epoch < 180; epoch++){
        for(int i = 0; i < examplesSubset.cols(); i++)
        {
            vae.Backpropagate(examplesSubset.col(i), settings, cost, epoch);
        }
        std::cout << epoch << std::endl;
        std::cout << cost << std::endl;
    }

    Eigen::VectorXd input = examplesSubset.col(0);
    Eigen::VectorXd prediction = vae.FeedForward(input);
    VectorToPPM((prediction * 255).cast<int>(), "output");
    VectorToPPM((input * 255).cast<int>(), "input");

    QParams qParams = vae.m_qParams;
    qParams.eps = Math::GenGaussianVector(qParams.mu.size(), 0, 1);
    Eigen::VectorXd latent = vae.CalculateLatent(qParams);
    Eigen::VectorXd prediction2 = vae.m_decoder.Decode(latent);
    VectorToPPM((prediction2 * 255).cast<int>(), "output2");
}
