/*
 * main_mnist.cpp
 * ==============
 * Application 3: handwritten digit classification (10-class).
 *
 * Uses sklearn_cpp::linear_model::LogisticRegression on data/mnist_micro.csv.
 * Owned by Person C as the integration test for the multi-class path.
 *
 * Build (from project root):
 *     g++ -std=c++17 -O2 -I ./include examples/main_mnist.cpp -o mnist.out
 *
 * Run:
 *     ./mnist.out
 *
 * Note on performance:
 *     mnist_micro.csv has 3000 samples x 784 features.  Training will be
 *     noticeably slower than the other two applications.  Start with
 *     n_iterations = 500 while debugging, then push higher for accuracy.
 *
 * Expected benchmark from Python sklearn:
 *     ~85-90% accuracy on mnist_micro (3000 samples)
 */

#include <iostream>
#include <vector>

#include "sklearn_cpp/utils/csv_reader.hpp"
#include "sklearn_cpp/linear_model/logistic_regression.hpp"

int main()
{
    // ------------------------------------------------------------------
    // 1. Load the dataset.
    //    mnist_micro.csv has NO header row.
    //    Features (784): pixel values, scaled to [0, 1]
    //    Label    (1):   digit class 0..9  (last column)
    // ------------------------------------------------------------------
    std::vector<std::vector<double>> X;
    std::vector<double>              y;

    std::cout << "Loading mnist_micro.csv (this may take a moment)...\n";

    try {
        sklearn_cpp::utils::CSVReader::load(
            "./data/mnist_micro.csv", /*has_header=*/false, X, y);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Loaded mnist_micro.csv\n"
              << "  Samples  : " << X.size()    << "\n"
              << "  Features : " << X[0].size() << "\n"
              << "  Classes  : 10 (digits 0-9)\n\n";

    // ------------------------------------------------------------------
    // 2. Train the multi-class classifier.
    //    fit() detects K=10 and dispatches to fit_multiclass_.
    // ------------------------------------------------------------------
    sklearn_cpp::linear_model::LogisticRegression model(
        /*learning_rate=*/0.1,
        /*n_iterations=*/500,
        /*lambda=*/0.01);

    std::cout << "Training LogisticRegression (multi-class)...\n";
    model.fit(X, y);

    // ------------------------------------------------------------------
    // 3. Evaluate accuracy.
    // ------------------------------------------------------------------
    const double acc = model.score(X, y);
    std::cout << "Accuracy (training set): " << acc * 100.0 << "%\n\n";

    // ------------------------------------------------------------------
    // 4. Demo: predict a few individual samples to show argmax works.
    // ------------------------------------------------------------------
    std::cout << "Sample predictions:\n";
    for (int idx : {0, 100, 500, 1500, 2500}) {
        if (static_cast<std::size_t>(idx) >= X.size()) break;
        const std::vector<std::vector<double>> one_sample = { X[idx] };
        const int pred = model.predict(one_sample)[0];
        std::cout << "  sample " << idx
                  << "  true=" << static_cast<int>(y[idx])
                  << "  pred=" << pred << "\n";
    }

    return 0;
}
