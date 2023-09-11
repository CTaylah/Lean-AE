#include "NeuralNetwork.h"
#include <iostream>
#include <ctime>

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

    double cost = 0;

    NeuralNetwork neuralNetwork({784, 16, 16, 10});
    for(int e = 0; e < 24; ++e){
        for(int i = 0; i < 4000; ++i){
            int index = rand() % 60000;
            Eigen::VectorXd column = examplesDouble.col(index);
            Eigen::VectorXd column2 = labelsDouble.col(index);

            neuralNetwork.Backpropagate(column, column2, 0.0008, cost);
        }
        std::cout << "Epoch: " << e << " Cost: " << cost << std::endl;
    }

    int index = rand() % 60000;
    std::cout << neuralNetwork.GetPrediction(examplesDouble.col(index)) << std::endl;
    int answer;
    for(int i = 0; i < 10; ++i){
        if(labelsDouble(i, index) == 1){
            answer = i;
        }
    }
    std::cout << "Index: " << index << std::endl;
    std::cout << "Answer: " << answer << std::endl;

}