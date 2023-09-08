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