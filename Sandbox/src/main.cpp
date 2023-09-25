#include "NeuralNetwork.h"
#include "Data.h"
#include <iostream>
#include <ctime>
#include <omp.h>

int main(int argc, char** argv){

    srand(time(NULL));

    Eigen::MatrixXi examples(784, 60000);
    Eigen::MatrixXi labels(10, 60000);

    Eigen::Ref<Eigen::MatrixXi> examplesRef = examples; 
    Eigen::Ref<Eigen::MatrixXi> labelsRef = labels;
    
    bool success = ReadMNISTImages("MNIST/training_images/train-images.idx3-ubyte", examplesRef);
    bool success2 = ReadMNISTLabels("MNIST/training_labels/train-labels.idx1-ubyte", labelsRef);
    if(!success || !success2){
        std::cout << "Failed to read MNIST data" << std::endl;
        return 1;
    }

    Eigen::MatrixXd examplesDouble = examplesRef.cast<double>();
    Eigen::MatrixXd labelsDouble = labelsRef.cast<double>();
    examplesDouble = examplesDouble / 255.0;


    NeuralNetwork neuralNetwork({784, 256, 256, 784});
    TrainingSettings settings(0.0115, 10, omp_get_num_threads());
    neuralNetwork.Train(settings, examplesDouble, examplesDouble);

    int index = rand() % 30000;
    index += 30000;
    std::cout << "index: " << index << std::endl;

    double meanSquaredError = Math::MeanSquaredError(neuralNetwork.GetPrediction(examplesDouble.col(index)), examplesDouble.col(index));
    std::cout << meanSquaredError << std::endl;
    int answer;
    for(int i = 0; i < 10; ++i){
        if(labelsDouble(i, index) == 1){
            answer = i;
        }
    }

    Eigen::VectorXd prediction = neuralNetwork.GetPrediction(examplesDouble.col(index));
    VectorToPPM((prediction * 255).cast<int>(), "Prediction");
    VectorToPPM((examplesDouble.col(index) * 255).cast<int>(), "Input");

}
