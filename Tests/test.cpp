
#ifdef DEBUG_TEST

#include "gtest/gtest.h"

#include "MathTest.hpp"
#include "LayerTest.hpp"
#include "NeuralNetworkTest.hpp"
#include "Math.h"

#include <iostream>
#include <time.h>


#endif
int main(int argc, char** argv){


    #ifdef DEBUG_TEST

    srand((unsigned) time(0));
    ::testing::InitGoogleTest();
    auto a = RUN_ALL_TESTS();
    
    #endif
    return 0;

}
