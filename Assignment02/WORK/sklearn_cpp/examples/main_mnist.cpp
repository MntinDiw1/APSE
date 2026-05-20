#include <iostream>
#include <vector>

#include "sklearn_cpp/utils/csv_reader.hpp"
#include "sklearn_cpp/linear_model/logistic_regression.hpp"

// Compile (from project root):
//   g++ -std=c++17 -O2 -I ./include examples/main_mnist.cpp -o mnist.out
// Run:
//   ./mnist.out

int main() {
    // Load the MNIST micro dataset (no header row)
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    std::cout << "Loading data...\n";
    sklearn_cpp::utils::CSVReader::load("./data/mnist_micro.csv", false, X, y);

    std::cout << "Loaded " << X.size() << " samples with " 
              << X[0].size() << " features\n";

    // Train multi-class logistic regression - fit() detects 10 classes and runs the softmax path
    // Using fewer iterations than the default because the dataset is much bigger
    sklearn_cpp::linear_model::LogisticRegression model(0.1, 500, 0.01);
    std::cout << "Training...\n";
    model.fit(X, y);

    // Report accuracy on the training set
    double accuracy = model.score(X, y);
    std::cout << "Accuracy: " << accuracy * 100 << "%\n";

    return 0;
}