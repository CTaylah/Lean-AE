#include "Eigen/Dense"

#include <iostream>
#include <fstream>


//Stochastic Gradient Descent
struct TrainingSettings{
    TrainingSettings(double learningRate, int epochs, int batchSize, const Eigen::VectorXd& trainingData, const Eigen::VectorXd& labels) 
        : learningRate(learningRate), epochs(epochs), batchSize(batchSize), trainingData(trainingData), labels(labels) {}

    
    Eigen::VectorXd trainingData;
    Eigen::VectorXd labels;

    double learningRate;
    int epochs;
    int batchSize;
};

void ReadMNISTImages(const std::string& filename, Eigen::VectorXd& data, int& numImages, int& numPixels);
void ReadMNISTLabels(const std::string& filename, Eigen::VectorXd& labels, int& numLabels);