

#include "gtest/gtest.h"

#include <iostream>
#include <time.h>

#include "LayerTest.hpp"
#include "NeuronTest.hpp"
#include "MathTest.hpp"

int main(int argc, char** argv){
    srand((unsigned) time(0));
    ::testing::InitGoogleTest();
    auto a = RUN_ALL_TESTS();
    return 0;
}
