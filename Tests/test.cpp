

#include "gtest/gtest.h"

#include "LayerTest.hpp"
#include "MathTest.hpp"
#include "NeuralNetworkTest.hpp"
#include "Math.h"

#include <iostream>
#include <time.h>


int main(int argc, char** argv){



    srand((unsigned) time(0));
    ::testing::InitGoogleTest();
    auto a = RUN_ALL_TESTS();
    
    return 0;

}
