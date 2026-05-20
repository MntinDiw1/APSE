#include <iostream>
#include <vector>

#include "sklearn_cpp/utils/csv_reader.hpp"
#include "sklearn_cpp/linear_model/linear_regression.hpp"


// Compile (from project root):
//   g++ -std=c++17 -O2 -I ./include examples/main_concrete.cpp -o concrete.out
// Run:
//   ./concrete.out

int main() {
    // Load the concrete dataset (has a header row)
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    sklearn_cpp::utils::CSVReader::load("./data/concrete.csv", true, X, y);

    std::cout << "Loaded " << X.size() << " samples with " 
              << X[0].size() << " features\n";

    // Train linear regression using gradient descent
    sklearn_cpp::linear_model::LinearRegression model;
    model.fit(X, y);

    // Report the R squared score on the training set
    double r2 = model.score(X, y);
    std::cout << "R squared: " << r2 << std::endl;

    return 0;
}
