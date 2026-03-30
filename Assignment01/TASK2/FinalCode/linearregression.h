#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <iostream>
#include <string>      
#include <vector>
#include "coordinate.h"


class LinearRegression {
public:
    // Task 1: Analytical Solution (OLS)
    static void performLinearRegression(const std::vector<Coordinate>& data, 
                                         double& slope, double& intercept) {
        double sumX = 0.0, sumY = 0.0, sumXY = 0.0, sumX2 = 0.0; //this is initilising all of those like sigma values in the equation
        int n = data.size(); // Number of data points

        for (const auto& point : data) { // Loop through each data point to calculate the necessary summations
            sumX += point.x; //sigma x
            sumY += point.y; //sigma y
            sumXY += point.x * point.y; //sigma xy
            sumX2 += point.x * point.x; //sigma x^2
        }

        slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX); //this is the slope formula from equation 6
        intercept = (sumY - slope * sumX) / n; //this is the intercept formula from equation 5

        std::cout << "\n=== OLS Results ===\nSlope: " << slope << "\nIntercept: " << intercept << std::endl;
    }

    // Task 2: Iterative Solution (Gradient Descent)
    static void performGradientDescent(const std::vector<Coordinate>& data, 
                                        double& w, double& b, 
                                        double learning_rate, int epochs) {
        int n = data.size(); // Number of data points
        w = 0.0; b = 0.0; // Initialize parameters of slope and intercept

        for (int i = 0; i < epochs; ++i) { // Loop for the number of iterations (epochs)
            double dL_dw = 0.0; // error derivative with respect to slope
            double dL_db = 0.0; // error derivative with respect to intercept

            for (const auto& point : data) {  // Loop through each data point to calculate the predictions and errors
                double prediction = (w * point.x) + b; // Calculate the predicted y value using the current parameters
                double error = prediction - point.y; // Calculate the error (difference between predicted and actual y)
                
                // Derivatives from Equation 7 and 8
                dL_dw += error * point.x; 
                dL_db += error; 
            }

            // Update parameters: Subtract the gradient
            w -= (2.0 / n) * learning_rate * dL_dw; // Update slope
            b -= (2.0 / n) * learning_rate * dL_db; // Update intercept
        }
        
        std::cout << "\n=== GD Results ===\nSlope: " << w << "\nIntercept: " << b << std::endl;
    }
};

#endif // LINEARREGRESSION_H