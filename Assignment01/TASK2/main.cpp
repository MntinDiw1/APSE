// Use Eigen vectors to complete Task 2.

#include<iostream>
#include<vector>
#include<random>
#include<fstream> 
#include "linearregression.h"
#include "coordinate.h"
#include "dataset.h"




int main() {
    DataGenerator mainGen(20, 2.0, 5.0, 2.0);
    std::vector<Coordinate> dataset = mainGen.CreateData();

    double fitted_slope = 0.0, fitted_intercept = 0.0;
    // Calling via class scope
    LinearRegression::performLinearRegression(dataset, fitted_slope, fitted_intercept);

    double Gfitted_slope = 0.0, Gfitted_intercept = 0.0;
    // Task 2 implementation
    LinearRegression::performGradientDescent(dataset, Gfitted_slope, Gfitted_intercept, 0.001, 5000);

    std::ofstream outFile("LinearDataset.csv");
    if (outFile.is_open()) {
        outFile << "x,y,f_slope,f_intercept,Gfitted_slope,Gfitted_intercept\n";
        for (const auto& point : dataset) {
            outFile << point.x << "," << point.y << "," 
                    << fitted_slope << "," << fitted_intercept << "," 
                    << Gfitted_slope << "," << Gfitted_intercept << "\n";
        }
        outFile.close();
        std::cout << "\nLinearDataset.csv created successfully." << std::endl;
    }

    return 0;
}




    /*
    References:
    https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution.html
    
    */