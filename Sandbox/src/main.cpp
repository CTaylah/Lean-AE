#include "Model/NeuralNetwork.h"
#include "Common/Data.h"
#include "Testing/CrossValidation.h"

#include <iostream>
#include <ctime>
#include <omp.h>
#include <chrono>
int main(int argc, char** argv){

    srand(time(NULL));
    int num_threads = 6;
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
    Eigen::MatrixXd examplesSubset = examplesDouble.block(0, 0, examplesDouble.rows(), 3500);

    NeuralNetwork network1({784, 162, 784});
    TrainingSettings settings(70, 24, 0.00087);
    NeuralNetwork network2({784, 1, 784});


    std::vector<NeuralNetwork> networks = {network1, network2};

    auto start = std::chrono::high_resolution_clock::now();

    double cost = Testing::MonteCarloCV(network1, settings, examplesSubset, 0.7);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);
    std::cout << duration.count() << std::endl;

    std::cout << "Cost: " << cost << std::endl;
    //MonteCarloCV(network2, 0.8, examplesSubset, settings);
}
