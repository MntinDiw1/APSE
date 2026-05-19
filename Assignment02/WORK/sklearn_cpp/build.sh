#!/bin/bash
# build.sh
# ========
# Compile all three example programs in one go.
# Run this from the project root (the directory containing this script).
#
# Usage:
#     chmod +x build.sh    # only the first time
#     ./build.sh

set -e   # stop on first error

CXX="g++"
CXXFLAGS="-std=c++17 -O2 -Wall -Wextra"
INCLUDE="-I ./include"

mkdir -p ./bin

echo "Building concrete.out ..."
$CXX $CXXFLAGS $INCLUDE examples/main_concrete.cpp -o ./bin/concrete.out

echo "Building ecg.out ..."
$CXX $CXXFLAGS $INCLUDE examples/main_ecg.cpp -o ./bin/ecg.out

echo "Building mnist.out ..."
$CXX $CXXFLAGS $INCLUDE examples/main_mnist.cpp -o ./bin/mnist.out

echo ""
echo "All three executables built in ./bin/"
echo "Run them with:"
echo "    ./bin/concrete.out"
echo "    ./bin/ecg.out"
echo "    ./bin/mnist.out"
