#include "NeuralNetwork.h"
#include "Data.h"
#include "CrossValidation.h"

#include <iostream>
#include <ctime>
#include <omp.h>

int main(int argc, char** argv){

    srand(time(NULL));

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
    Eigen::MatrixXd examplesSubset = examplesDouble.block(0, 0, examplesDouble.rows(), 200);


    NeuralNetwork neuralNetwork({784, 392, 784});
    TrainingSettings settings(120, 40, 0.00085);
    
    std::cout << "here" << std::endl;
    //neuralNetwork.Train(settings, examplesSubset, examplesSubset);
    MonteCarloCV(neuralNetwork, 0.9, examplesSubset, settings);

}
