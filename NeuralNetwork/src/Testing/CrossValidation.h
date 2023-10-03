#pragma once

#include "Model/NeuralNetwork.h"
#include "Common/Data.h"
#include <omp.h>


namespace Testing{
//Tests a network on a testing set, returns the average cost
double Test(NeuralNetwork network, const Eigen::MatrixXd& testingSet, bool verbose=false);

//trainingPercent specifies what percentage of the data will be used for training
//Returns the average cost of the network on the testing set
double MonteCarloCV(NeuralNetwork network, TrainingSettings settings, Eigen::MatrixXd& examples, double trainingPercent);

//Uses two threads, training and testing each network on a different thread
std::vector<double> Compare(NeuralNetwork network1, TrainingSettings settings1, NeuralNetwork network2, TrainingSettings settings2, 
    Eigen::MatrixXd& dataSet, double trainingPercent);

std::vector<double> Compare(std::vector<NeuralNetwork> networks, TrainingSettings settings, 
    Eigen::MatrixXd& dataSet, double trainingPercent);

void GridSearch(std::vector<NeuralNetwork> networks, std::vector<TrainingSettings> settings, 
    Eigen::MatrixXd& dataSet, double trainingPercent );

}

