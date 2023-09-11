#pragma once

#include "Eigen/Dense"

#include <iostream>
#include <string>
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

[[nodiscard]] bool ReadMNISTImages(const std::string& filename, Eigen::Ref<Eigen::MatrixXi>& data);
[[nodiscard]] bool ReadMNISTLabels(const std::string& filename, Eigen::Ref<Eigen::MatrixXi>& labels);