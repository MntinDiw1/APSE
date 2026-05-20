/*
 * main_concrete.cpp
 * =================
 * Application 1: predict concrete compressive strength.
 *
 * Uses sklearn_cpp::linear_model::LinearRegression on data/concrete.csv.
 * Owned by Person A as the integration test for the LinearRegression class.
 *
 * Build (from project root):
 *     g++ -std=c++17 -O2 -I ./include examples/main_concrete.cpp -o concrete.out
 *
 * Run:
 *     ./concrete.out
 *
 * Expected benchmark from Python sklearn (full dataset, no split):
 *     R^2 score around 0.6
 */

#include <iostream>
#include <vector>

#include "sklearn_cpp/utils/csv_reader.hpp"
#include "sklearn_cpp/linear_model/linear_regression.hpp"

int main()
{
    // ------------------------------------------------------------------
    // 1. Load the dataset.
    //    concrete.csv has a header row, so has_header = true.
    //    Features (8): cement, blast furnace slag, fly ash, water,
    //                  superplasticizer, coarse aggregate, fine aggregate, age
    //    Label   (1):  compressive strength (MPa)
    // ------------------------------------------------------------------
    std::vector<std::vector<double>> X;
    std::vector<double>              y;

    try {
        sklearn_cpp::utils::CSVReader::load(
            "./data/concrete.csv", /*has_header=*/true, X, y);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Loaded concrete.csv\n"
              << "  Samples  : " << X.size()    << "\n"
              << "  Features : " << X[0].size() << "\n\n";

    // ------------------------------------------------------------------
    // 2. Train the model.
    //    A very small learning rate is needed because the input features
    //    span 2-3 orders of magnitude (e.g. cement ~ 500, age ~ 28).
    // ------------------------------------------------------------------
    sklearn_cpp::linear_model::LinearRegression model;

    std::cout << "Training LinearRegression...\n";
    model.fit(X, y);

    // ------------------------------------------------------------------
    // 3. Evaluate on the training set.
    // ------------------------------------------------------------------
    const double r2 = model.score(X, y);
    std::cout << "R^2 score (training set): " << r2 << "\n";

    return 0;
}
