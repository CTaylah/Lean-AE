#pragma once

#define private public
#include "NeuralNetwork.h"
#undef private

#include "gtest/gtest.h"

#include <vector>


TEST(NeuralNetwork, Constructor)
{
    std::vector<unsigned int> neuronsPerLayer = {2, 5, 64, 32};
    Topology top(12, neuronsPerLayer);
    NeuralNetwork network(top);
    

    EXPECT_EQ(top.neuronsPerLayer[2], network.m_layers[2].m_size);
    EXPECT_EQ(top.inputSize, network.m_layers[0].m_size);


}