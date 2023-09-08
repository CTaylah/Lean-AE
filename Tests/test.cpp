
#ifdef DEBUG_TEST

#include "gtest/gtest.h"

#include <iostream>
#include <time.h>

#include "LayerTest.hpp"
#include "NeuronTest.hpp"
#include "MathTest.hpp"

#endif
int main(int argc, char** argv){

    #ifdef DEBUG_TEST

    srand((unsigned) time(0));
    ::testing::InitGoogleTest();
    auto a = RUN_ALL_TESTS();

    #endif
}
