#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <iostream>
#include <string>      
#include <vector>
#include "coordinate.h"

// Class to perform linear regression using both OLS and GD methods
class LinearRegression {
public:
<<<<<<< HEAD
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
=======
    // Perform Ordinary Least Squares Linear Regression
    static void performLinearRegression(const std::vector<coordinate>& data, 
                                        std::vector<double>& w, double& b) {
        int n = data.size();
        if (n == 0) return;
        int k = data[0].x.size();

        std::vector<std::vector<double>> X(n, std::vector<double>(k + 1, 1.0));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < k; ++j) {
                X[i][j + 1] = data[i].x[j];
            }
        }

        std::vector<std::vector<double>> XtX(k + 1, std::vector<double>(k + 1, 0.0));
        std::vector<double> XtY(k + 1, 0.0);
>>>>>>> 09bfa48f5eedc987a86e41f668aa73c19597f2ee

        for (int i = 0; i < k + 1; ++i) {
            for (int j = 0; j < k + 1; ++j) {
                for (int d = 0; d < n; ++d) {
                    XtX[i][j] += X[d][i] * X[d][j];
                }
            }
            for (int d = 0; d < n; ++d) {
                XtY[i] += X[d][i] * data[d].y;
            }
        }

        std::vector<std::vector<double>> aug(k + 1, std::vector<double>(k + 2, 0.0));
        for (int i = 0; i < k + 1; ++i) {
            for (int j = 0; j < k + 1; ++j) {
                aug[i][j] = XtX[i][j];
            }
            aug[i][k + 1] = XtY[i];
        }

        for (int i = 0; i < k + 1; ++i) {
            double pivot = aug[i][i];
            for (int j = 0; j <= k + 1; ++j) {
                aug[i][j] /= pivot;
            }
            for (int j = 0; j < k + 1; ++j) {
                if (i != j) {
                    double factor = aug[j][i];
                    for (int l = 0; l <= k + 1; ++l) {
                        aug[j][l] -= factor * aug[i][l];
                    }
                }
            }
        }

        b = aug[0][k + 1];
        w.resize(k);
        std::cout << "\n=== OLS Results ===\nIntercept: " << b << "\nSlopes: ";
        for (int i = 0; i < k; ++i) {
            w[i] = aug[i + 1][k + 1];
            std::cout << w[i] << " ";
        }
        std::cout << std::endl;
    }

    // Perform Gradient Descent Linear Regression
    static void performGradientDescent(const std::vector<coordinate>& data, 
                                        std::vector<double>& w, double& b, 
                                        double learning_rate, int epochs) {
<<<<<<< HEAD
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
=======
        int n = data.size();
        if (n == 0) return;
        int k = data[0].x.size();
        w.assign(k, 0.0); 
        b = 0.0; 

        for (int i = 0; i < epochs; ++i) {
            std::vector<double> dL_dw(k, 0.0);
            double dL_db = 0.0;

            for (const auto& point : data) {
                double prediction = b;
                for (int j = 0; j < k; ++j) {
                    prediction += w[j] * point.x[j];
                }
                double error = prediction - point.y;
                
                for (int j = 0; j < k; ++j) {
                    dL_dw[j] += error * point.x[j];
                }
                dL_db += error;
            }

            for (int j = 0; j < k; ++j) {
                w[j] -= (2.0 / n) * learning_rate * dL_dw[j];
            }
            b -= (2.0 / n) * learning_rate * dL_db;
>>>>>>> 09bfa48f5eedc987a86e41f668aa73c19597f2ee
        }
        
        std::cout << "\n=== GD Results ===\nIntercept: " << b << "\nSlopes: ";
        for (double val : w) std::cout << val << " ";
        std::cout << std::endl;
    }
};

#endif // LINEARREGRESSION_H