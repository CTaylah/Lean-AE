#include "gtest/gtest.h"
#include "Math.h"



TEST(Math, ReLU)
{
    EXPECT_EQ(Math::ReLU(0), 0);
    EXPECT_EQ(Math::ReLU(1), 1);
    EXPECT_EQ(Math::ReLU(-1), 0);
    EXPECT_EQ(Math::ReLU(100), 100);
    EXPECT_EQ(Math::ReLU(-100), 0);
}

TEST(Math, ReLUDerivative)
{
    EXPECT_EQ(Math::ReLUDerivative(0), 0);
    EXPECT_EQ(Math::ReLUDerivative(1), 1);
    EXPECT_EQ(Math::ReLUDerivative(-1), 0);
    EXPECT_EQ(Math::ReLUDerivative(100), 1);
    EXPECT_EQ(Math::ReLUDerivative(-100), 0);
}

TEST(Math, LeakyReLU)
{
    EXPECT_EQ(Math::LeakyReLU(0, 0.01), 0);
    EXPECT_EQ(Math::LeakyReLU(1, 0.01), 1);
    EXPECT_EQ(Math::LeakyReLU(-1, 0.01), -0.01);
    EXPECT_EQ(Math::LeakyReLU(100, 0.01), 100);
    EXPECT_EQ(Math::LeakyReLU(-100, 0.01), -1);
}

TEST(Math, LeakyReLUDerivative)
{
    EXPECT_EQ(Math::LeakyReLUDerivative(0, 0.01), 0.01);
    EXPECT_EQ(Math::LeakyReLUDerivative(1, 0.01), 1);
    EXPECT_EQ(Math::LeakyReLUDerivative(-1, 0.01), 0.01);
    EXPECT_EQ(Math::LeakyReLUDerivative(100, 0.01), 1);
    EXPECT_EQ(Math::LeakyReLUDerivative(-100, 0.01), 0.01);
}
