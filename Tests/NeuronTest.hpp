#include <gtest/gtest.h>

#define private public
#include "Neuron.h"
#undef private


TEST(Neuron, Constructor)
{
    int numberInputs = Math::RandomInt(1, 100);
    Neuron n(numberInputs);
    EXPECT_EQ(n.m_weights.size(), numberInputs);
}

TEST(Neuron, GetActivation)
{
    int numberInputs = Math::RandomInt(1, 100);
    Neuron n(numberInputs);
    Eigen::VectorXd inputs = Eigen::VectorXd::Random(numberInputs);
    double activation = n.GetActivation(inputs);
    EXPECT_EQ(activation, Math::ReLU(n.m_weights.dot(inputs) + n.m_bias));
}