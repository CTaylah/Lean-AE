#pragma once

#define private public
#include "NeuralNetwork.h"
#undef private

#include "gtest/gtest.h"

#include <vector>


TEST(NeuralNetwork, Constructor)
{
    std::vector<int> top = {12, 2, 5, 64, 32};
    NeuralNetwork network(top);
    

    EXPECT_EQ(top[2], network.m_layers[2].m_size);
    EXPECT_EQ(top[0], network.m_layers[0].m_size);


}

TEST(NeuralNetwork, ForwardPropagation)
{
    NeuralNetwork network(std::vector<int>({3, 4, 5, 6}));
    Eigen::VectorXd inputs(3);
    inputs << 1, 1, 0;
    network.FeedForward(inputs);


}

TEST(NeuralNetwork, Backpropagation)
{
    NeuralNetwork network(std::vector<int>({1, 1, 1, 1}));
    Eigen::VectorXd inputs(1);
    inputs << 0;

    Eigen::VectorXd targets(1);
    targets << 1;

for(int i = 0; i < 10; i++){
    NeuralNetwork network(std::vector<int>({8, 7, 6}));
    Eigen::VectorXd inputs(8);
    inputs << 0.43, 0.9, 0.3, 0.034, 0.12, 0.3232, 0.1, 0.23;

    Eigen::VectorXd targets(6);
    targets << .98, 0, 0, 0, 0, 0;


    for(int i = 0; i < 50000; i++)
        network.Backpropagate(inputs, targets, 0.05);
    
    std::cout << network.GetPrediction(inputs) << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    }
}

