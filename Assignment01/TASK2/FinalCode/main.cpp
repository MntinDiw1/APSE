// Use Eigen vectors to complete Task 2.

#include<iostream>
#include<vector>
#include<random>
#include<fstream> 
#include "linearregression.h"  // Joel's code for linear regression NO EIGEN VECTORS, updated by Saad to do linear regression with two x values and one y value
#include "coordinate.h" // Joel's code for the Coordinate struct, which bundles x and y values together, updated by Saad to have two x values for the multipoint dataset
#include "dataset.h" // Will's code for dataset generation, Updated by Saad to generate a multipoint dataset with two x values and one y value
#include "plot.h" // Saad's code for plotting the data and regression lines

int main() {
    DataGenerator mainGen;

    // TASK 2.1: 1D LINEAR REGRESSION (Single Variable)
    std::cout << "========== 1D LINEAR REGRESSION ==========" << std::endl;
    double true_w1_1d = 3.0;
    double true_b_1d = 4.0;
    std::cout << "Hypothetical/True Weights -> Intercept: " << true_b_1d << " | Slope: " << true_w1_1d << std::endl;
    // Generate a single x value dataset with the specified parameters
    mainGen.singlepointGen(20, true_w1_1d, true_b_1d, 2.0); 
    std::vector<coordinate> dataset_1d = mainGen.CreateSinglepointData(); 
    // Perform OLS and GD on the 1D dataset
    std::vector<double> w_ols_1d;
    double b_ols_1d = 0.0;
    LinearRegression::performLinearRegression(dataset_1d, w_ols_1d, b_ols_1d);

    std::vector<double> w_gd_1d;
    double b_gd_1d = 0.0;
    LinearRegression::performGradientDescent(dataset_1d, w_gd_1d, b_gd_1d, 0.001, 5000);

    // Print delta between true and estimated weights and intercept for 1D OLS and GD
    std::cout << "\n=== 1D OLS Weight Differences ===\n";
    std::cout << "Delta Intercept: " << std::abs(true_b_1d - b_ols_1d) << "\n";
    std::cout << "Delta w1: " << std::abs(true_w1_1d - w_ols_1d[0]) << "\n";

    // Plotting the 1D results
    Plot::plotData1D(dataset_1d, w_ols_1d, b_ols_1d, w_gd_1d, b_gd_1d);

    // ==========================================
    // TASK 2.2: 2D LINEAR REGRESSION (Multiple Variables)
    // ==========================================
    std::cout << "\n========== 2D LINEAR REGRESSION ==========" << std::endl;
    double true_w1_2d = 2.0;
    double true_w2_2d = 1.0;
    double true_b_2d = 5.0;
    std::cout << "Hypothetical/True Weights -> Intercept: " << true_b_2d << " | Slopes: " << true_w1_2d << " " << true_w2_2d << std::endl;

    mainGen.multipointGen(20, true_w1_2d, true_w2_2d, true_b_2d, 2.0);  
    std::vector<coordinate> dataset_2d = mainGen.CreateMultipointData(); 

    std::vector<double> w_ols_2d;
    double b_ols_2d = 0.0;
    LinearRegression::performLinearRegression(dataset_2d, w_ols_2d, b_ols_2d);

    std::vector<double> w_gd_2d;
    double b_gd_2d = 0.0;
    LinearRegression::performGradientDescent(dataset_2d, w_gd_2d, b_gd_2d, 0.001, 5000);

    // Print delta between true and estimated weights and intercept for 2D OLS and GD
    std::cout << "\n=== 2D OLS Weight Differences ===\n";
    std::cout << "Delta Intercept: " << std::abs(true_b_2d - b_ols_2d) << "\n";
    for (size_t i = 0; i < w_ols_2d.size(); ++i) {
        std::cout << "Delta w" << i + 1 << ": " << std::abs((i == 0 ? true_w1_2d : true_w2_2d) - w_ols_2d[i]) << "\n";
    }

    std::cout << "\n=== 2D GD Weight Differences ===\n";
    std::cout << "Delta Intercept: " << std::abs(true_b_2d - b_gd_2d) << "\n";
    for (size_t i = 0; i < w_gd_2d.size(); ++i) {
        std::cout << "Delta w" << i + 1 << ": " << std::abs((i == 0 ? true_w1_2d : true_w2_2d) - w_gd_2d[i]) << "\n";
    }

    // Plotting the 2D results
    Plot::plotData(dataset_2d, w_ols_2d, b_ols_2d, w_gd_2d, b_gd_2d);

    
    // NOTE: If you want to fix the CSV export, you need to loop through dataset_2d
    // and extract point.x[0] and point.x[1] individually, rather than just point.x.
    
    return 0;
}

    /*
    References:
    https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution.html
    
    */