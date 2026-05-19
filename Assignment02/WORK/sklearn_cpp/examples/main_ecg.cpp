/*
 * main_ecg.cpp
 * ============
 * Application 2: ECG anomaly detection (binary classification).
 *
 * Uses sklearn_cpp::linear_model::LogisticRegression on data/ecg.csv.
 * Owned by Person B as the integration test for the binary path.
 *
 * Build (from project root):
 *     g++ -std=c++17 -O2 -I ./include examples/main_ecg.cpp -o ecg.out
 *
 * Run:
 *     ./ecg.out
 *
 * Expected benchmark from Python sklearn:
 *     ~99% accuracy on the training set
 */

#include <iostream>
#include <vector>

#include "sklearn_cpp/utils/csv_reader.hpp"
#include "sklearn_cpp/linear_model/logistic_regression.hpp"

int main()
{
    // ------------------------------------------------------------------
    // 1. Load the dataset.
    //    ecg.csv has NO header row.
    //    Features (140): ECG signal samples per row
    //    Label   (1):    0 = abnormal, 1 = normal  (last column)
    // ------------------------------------------------------------------
    std::vector<std::vector<double>> X;
    std::vector<double>              y;

    try {
        sklearn_cpp::utils::CSVReader::load(
            "./data/ecg.csv", /*has_header=*/false, X, y);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Loaded ecg.csv\n"
              << "  Samples  : " << X.size()    << "\n"
              << "  Features : " << X[0].size() << "\n"
              << "  Classes  : 2 (0 = abnormal, 1 = normal)\n\n";

    // ------------------------------------------------------------------
    // 2. Train the binary classifier.
    //    fit() detects K=2 and dispatches to fit_binary_.
    // ------------------------------------------------------------------
    sklearn_cpp::linear_model::LogisticRegression model(
        /*learning_rate=*/0.1,
        /*n_iterations=*/1000,
        /*lambda=*/0.01);

    std::cout << "Training LogisticRegression (binary)...\n";
    model.fit(X, y);

    // ------------------------------------------------------------------
    // 3. Evaluate accuracy.
    // ------------------------------------------------------------------
    const double acc = model.score(X, y);
    std::cout << "Accuracy (training set): " << acc * 100.0 << "%\n";

    return 0;
}
