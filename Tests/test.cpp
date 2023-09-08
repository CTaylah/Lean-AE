



#include "gtest/gtest.h"

#include "LayerTest.hpp"
#include "NeuralNetworkTest.hpp"
#include "Math.h"

#include <iostream>
#include <time.h>


TEST(Math, RandomDouble)
{
    for(int i =0; i < 1000; i++)
    {
        double a = Math::RandomDouble(-0.1, 0.1);
        EXPECT_GE(a, -0.1);
        EXPECT_LE(a, 0.1);
    }
}

int main(int argc, char** argv){
    srand((unsigned) time(0));
    ::testing::InitGoogleTest();
    auto a = RUN_ALL_TESTS();
    return 0;
}
