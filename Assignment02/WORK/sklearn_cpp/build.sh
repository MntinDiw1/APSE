#!/bin/bash

#bash script to compile all three programs in one go.
# - - - run with: ./build.sh - - -

#creates a bin directory to store compiled files if it doesnt exist already.
mkdir -p ./bin

#files are overwritten if they already exist for simplicity.
echo "- - - Building concrete.out - - -"
g++ -I ./include examples/main_concrete.cpp -o ./bin/concrete.out

echo "- - - Building ecg.out - - -"
g++ -I ./include examples/main_ecg.cpp -o ./bin/ecg.out

echo "- - - Building mnist.out - - -"
g++ -I ./include examples/main_mnist.cpp -o ./bin/mnist.out

echo "Done. Run the following files:"
echo "  ./bin/concrete.out"
echo "  ./bin/ecg.out"
echo "  ./bin/mnist.out"