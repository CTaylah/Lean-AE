#!/bin/bash

# Set default build type to debug
build_type="debug"

# Check if argument is provided
if [[ $# -gt 0 ]]; then
  # Check if argument is -d or -r
  if [[ $1 == "-d" ]]; then
    build_type="debug"
  elif [[ $1 == "-r" ]]; then
    build_type="release"
  else
    echo "Invalid argument. Usage: ./build.sh [-d|-r]"
    exit 1
  fi
fi


# Generate makefiles using premake
./premake/premake5 gmake2

# Hack to enable openmp
# sed -i '33i ALL_LDFLAGS += -fopenmp' build/Sandbox/Makefile
# sed -i '20i ALL_LDFLAGS += -fopenmp' build/NeuralNetwork/Makefile
make config=$build_type
