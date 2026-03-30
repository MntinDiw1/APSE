#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include <iostream>
#include <string>      
#include <vector>
#include "coordinate.h"

// Class to perform linear regression using both OLS and GD methods
class LinearRegression {
public:
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
        }
        
        std::cout << "\n=== GD Results ===\nIntercept: " << b << "\nSlopes: ";
        for (double val : w) std::cout << val << " ";
        std::cout << std::endl;
    }
};

#endif // LINEARREGRESSION_H