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
        double sumX = 0.0, sumY = 0.0, sumXY = 0.0, sumX2 = 0.0; 
        int n = data.size();

        for (const auto& point : data) {
            sumX += point.x;
            sumY += point.y;
            sumXY += point.x * point.y;
            sumX2 += point.x * point.x;
        }

        slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        intercept = (sumY - slope * sumX) / n;

        std::cout << "\n=== OLS Results ===\nSlope: " << slope << "\nIntercept: " << intercept << std::endl;
    }

    // Task 2: Iterative Solution (Gradient Descent)
    static void performGradientDescent(const std::vector<Coordinate>& data, 
                                        double& w, double& b, 
                                        double learning_rate, int epochs) {
        int n = data.size();
        w = 0.0; b = 0.0; 

        for (int i = 0; i < epochs; ++i) {
            double dL_dw = 0.0;
            double dL_db = 0.0;

            for (const auto& point : data) {
                double prediction = (w * point.x) + b;
                double error = prediction - point.y;
                
                // Derivatives from Equation 7 and 8
                dL_dw += error * point.x;
                dL_db += error;
            }

            // Update parameters: Subtract the gradient
            w -= (2.0 / n) * learning_rate * dL_dw;
            b -= (2.0 / n) * learning_rate * dL_db;
        }
        
        std::cout << "\n=== GD Results ===\nSlope: " << w << "\nIntercept: " << b << std::endl;
    }
};

#endif // LINEARREGRESSION_H