#include <iostream>
#include <vector>

#include "sklearn_cpp/utils/csv_reader.hpp"
#include "sklearn_cpp/linear_model/logistic_regression.hpp"

// Compile (from project root):
//   g++ -std=c++17 -O2 -I ./include examples/main_ecg.cpp -o ecg.out
// Run:
//   ./ecg.out

int main() {
    // Load the ECG dataset (no header row)
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    sklearn_cpp::utils::CSVReader::load("./data/ecg.csv", false, X, y);

    std::cout << "Loaded " << X.size() << " samples with " 
              << X[0].size() << " features\n";

    // Train binary logistic regression - fit() detects 2 classes and runs the binary path
    sklearn_cpp::linear_model::LogisticRegression model;
    model.fit(X, y);

    // Report accuracy on the training set
    double accuracy = model.score(X, y);
    std::cout << "Accuracy: " << accuracy * 100 << "%\n";

    return 0;
}
