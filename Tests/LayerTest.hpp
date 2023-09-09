#pragma once

#include "gtest/gtest.h"

#define private public
#include "Layer.h"
#undef private


TEST(Layer, Constructor)
{

    int numberInputs = Math::RandomInt(1, 100);
    int numberNeurons = Math::RandomInt(1, 100);

    Layer l(numberInputs, numberNeurons);
    EXPECT_EQ(l.m_weights.rows(), numberNeurons);
    EXPECT_EQ(l.m_weights.cols(), numberInputs);
}

// Hand calculated values
TEST(LayerTest, FeedForward) {
    Layer layerUnderTest(3,4);
    layerUnderTest.m_weights << 0.2, -0.3, 0.5,
                                -0.4, 0.7, -0.2,
                                0.1, 0.2, -0.3,
                                0.5, -0.1, -0.2;
    
    layerUnderTest.m_biases << 0.1, -0.2, 0.3, -0.1;
    Eigen::VectorXd input(3);
    input << 0.5, -0.2, 0.8;

    Eigen::VectorXd activations = layerUnderTest.FeedForward(input);

    Eigen::VectorXd expectedActivations(4);
    expectedActivations << 0.66, 0.0, 0.07, 0.01;  // Expected values based on specific weights and biases

    // Check if the activations are approximately equal to the expected values
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(activations(i), expectedActivations(i), 1e-3);  // Tolerance used for approximate equality
    }

}