#pragma once

#include "Model/NeuralNetwork.h"
#include "Common/Data.h"
#include <omp.h>



void Test(NeuralNetwork network, const Eigen::MatrixXd& testingSet);

void MonteCarloCV(NeuralNetwork network, double trainingPercent, Eigen::MatrixXd& examples, TrainingSettings settings);

enum class GridSearchType{
    LearningRate,
    Momentum,
    
};

void GridSearch(NeuralNetwork network, double trainingPercent, Eigen::MatrixXd& examples, TrainingSettings settings, std::vector<double> learningRates, std::vector<double> momentums);
