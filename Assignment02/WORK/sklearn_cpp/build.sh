#!/bin/bash

# Shell script to compile all three programs in one go.
# Run from the project root with: ./build.sh

# Make the output folders if they don't exist already
mkdir -p ./bin
mkdir -p ./plots

# Python flags needed by the plotting library (matplotlibcpp embeds Python)
PYFLAGS=$(python3-config --cflags)
PYLIBS=$(python3-config --ldflags --embed)

echo "- - - Building concrete.out - - -"
g++ -I ./include $PYFLAGS examples/main_concrete.cpp -o ./bin/concrete.out $PYLIBS

echo "- - - Building ecg.out - - -"
g++ -I ./include $PYFLAGS examples/main_ecg.cpp -o ./bin/ecg.out $PYLIBS

echo "- - - Building mnist.out - - -"
g++ -I ./include $PYFLAGS examples/main_mnist.cpp -o ./bin/mnist.out $PYLIBS

echo "Done. Run the following files:"
echo "  ./bin/concrete.out"
echo "  ./bin/ecg.out"
echo "  ./bin/mnist.out"