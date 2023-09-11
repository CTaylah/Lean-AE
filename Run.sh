#!/bin/bash

# Check for argument "-d" or "-r"
if [[ $1 == "-d" ]]; then
  # Run debug build
  ./bin/Sandbox/Debug/Sandbox
elif [[ $1 == "-r" ]]; then
  # Run release build
  ./bin/Sandbox/Release/Sandbox
elif [[ $1 == "-t" ]]; then
  # Run release build
  ./bin/Tests/Debug/Tests
else
  # Invalid argument
  echo "Invalid argument. Usage: ./Run.sh [-d|-r|-t]"
  exit 1
fi