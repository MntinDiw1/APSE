#include <iostream>
#include <vector>

#include "sklearn_cpp/utils/csv_reader.hpp"
#include "sklearn_cpp/utils/visualize.hpp"
#include "sklearn_cpp/linear_model/linear_regression.hpp"

int main() {
    // Load the concrete dataset
    constexpr auto CSVPATH = "./data/concrete.csv"; // Path to the concrete CSV file (CHANGE THIS TO YOUR PATH)

    std::vector<std::vector<double>> X;
    std::vector<double> y;
    sklearn_cpp::utils::CSVReader::load(CSVPATH, true, X, y);

    std::cout << "Loaded " << X.size() << " samples with " << X[0].size() << " features" << std::endl;

    // Train linear regression using gradient descent
    sklearn_cpp::linear_model::LinearRegression model;
    model.fit(X, y);

    // Get predictions to feed the plotting helpers
    std::vector<double> y_pred = model.predict(X);

    // Report the R squared score on the training set
    double r2 = model.score(X, y);
    std::cout << "R squared: " << r2 << std::endl;

    // Plot actual values against predicted values - points on the diagonal mean a good fit
    sklearn_cpp::utils::Visualizer::plot_actual_vs_predicted(
        y, y_pred,
        "Concrete Strength - Actual vs Predicted",
        "./plots/concrete_actual_vs_predicted.png"); // CHANGE THIS TO YOUR DESIRED PATH

    // Plot residuals - should look like random noise around zero if the model is fitting well
    sklearn_cpp::utils::Visualizer::plot_residuals(
        y, y_pred,
        "Concrete Strength - Residuals",
        "./plots/concrete_residuals.png");  // CHANGE THIS TO YOUR DESIRED PATH

    return 0;
}