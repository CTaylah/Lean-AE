#include "NeuralNetwork.h"
#include "Data.h"
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


    NeuralNetwork neuralNetwork({784, 256, 120, 256, 784});
    TrainingSettings settings(25, omp_get_num_threads(), 0.0005);
    neuralNetwork.Train(settings, examplesDouble, examplesDouble);

    int index = rand() % 300;
    index = 9;
    // index += 30000;
    std::cout << "index: " << index << std::endl;

    double meanSquaredError = Math::MeanSquaredError(neuralNetwork.GetPrediction(examplesDouble.col(index)), examplesDouble.col(index));
    std::cout << meanSquaredError << std::endl;

    Eigen::VectorXd prediction = neuralNetwork.GetPrediction(examplesDouble.col(index));
    VectorToPPM((prediction * 255).cast<int>(), "Prediction");
    VectorToPPM((examplesDouble.col(index) * 255).cast<int>(), "Input");

}
