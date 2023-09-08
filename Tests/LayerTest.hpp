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
    Neuron* n = &l.m_neurons[0];

    EXPECT_EQ(l.m_weights.row(0).isApprox(n->m_weights.transpose()), true);
    EXPECT_EQ(l.m_weights.cols(), numberInputs);

}


