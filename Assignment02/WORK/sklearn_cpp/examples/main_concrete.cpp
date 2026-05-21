#include <iostream>
#include <vector>
#include <numeric>   // Required for std::iota
#include <random>    // Required for std::shuffle

#include "sklearn_cpp/utils/csv_reader.hpp"
#include "sklearn_cpp/utils/visualize.hpp"
#include "sklearn_cpp/linear_model/linear_regression.hpp"

int main() {
    // Load the concrete dataset
    constexpr auto TrainCSVPATH = "./data/concrete.csv"; // Path to the concrete CSV file
    constexpr auto TestCSVPATH = "";                     // Leave blank for automatic 80:20 split

    std::vector<std::vector<double>> X_train;
    std::vector<double> y_train;
    std::vector<std::vector<double>> X_test;
    std::vector<double> y_test;

    // Check if a separate test CSV was provided, if not split the training data into training and testing sets
    if (std::string(TestCSVPATH) != "") {
        // Load training and testing data from separate CSV files
        // Note: Using 'true' because concrete.csv has a header row
        sklearn_cpp::utils::CSVReader::load(TrainCSVPATH, true, X_train, y_train);
        std::cout << "Loaded Training data: " << X_train.size() << " samples with " << X_train[0].size() << " features" << std::endl;
        
        sklearn_cpp::utils::CSVReader::load(TestCSVPATH, true, X_test, y_test);
        std::cout << "Loaded Test data: " << X_test.size() << " samples with " << X_test[0].size() << " features" << std::endl;
    } 
    // If there is no separate test data, perform a randomized 80:20 split:
    else {
        std::cout << "No separate test CSV provided, using randomized 80:20 split" << std::endl;
        
        // Load entire CSV file
        std::vector<std::vector<double>> X;
        std::vector<double> y;
        sklearn_cpp::utils::CSVReader::load(TrainCSVPATH, true, X, y);
        std::cout << "Loaded data: " << X.size() << " samples with " << X[0].size() << " features" << std::endl;
        
        const std::size_t n_samples = X.size();
        const std::size_t n_train = static_cast<std::size_t>(n_samples * 0.8);
        
        // Create a vector of indices
        std::vector<std::size_t> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        
        // Shuffle the indices pseudo-randomly
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
        
        // Distribute the data using the shuffled indices
        for (std::size_t i = 0; i < n_train; ++i) {
            X_train.push_back(X[indices[i]]);
            y_train.push_back(y[indices[i]]);
        }
        
        for (std::size_t i = n_train; i < n_samples; ++i) {
            X_test.push_back(X[indices[i]]);
            y_test.push_back(y[indices[i]]);
        }
        
        std::cout << "Training set: " << X_train.size() << " samples with " << X_train[0].size() << " features" << std::endl;
        std::cout << "Test set: " << X_test.size() << " samples with " << X_test[0].size() << " features" << std::endl;
    }

    // Train linear regression using gradient descent on the TRAINING set
    sklearn_cpp::linear_model::LinearRegression model;
    model.fit(X_train, y_train);

    // Get predictions on the unseen TEST set to feed the plotting helpers
    std::vector<double> y_pred = model.predict(X_test);

    // Report the R squared score on the TEST set
    double r2 = model.score(X_test, y_test);
    std::cout << "R squared (Test Set): " << r2 << std::endl;

    // Plot actual values against predicted values for the TEST set
    sklearn_cpp::utils::Visualizer::plot_actual_vs_predicted(
        y_test, y_pred,
        "Concrete Strength - Actual vs Predicted",
        "./plots/concrete_actual_vs_predicted.png"); // CHANGE THIS TO YOUR DESIRED PATH

    // Plot residuals for the TEST set
    sklearn_cpp::utils::Visualizer::plot_residuals(
        y_test, y_pred,
        "Concrete Strength - Residuals",
        "./plots/concrete_residuals.png");  // CHANGE THIS TO YOUR DESIRED PATH

    return 0;
}