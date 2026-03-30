#ifndef PLOT_H
#define PLOT_H
#define WITHOUT_NUMPY

#include <vector>
#include <iostream>
#include "coordinate.h"
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

class Plot { // Class to handle plotting of data and regression lines
public:
    static void plotData(const std::vector<Coordinate>& dataset, double fitted_slope, double fitted_intercept, double Gfitted_slope, double Gfitted_intercept) {
        std::vector<double> x_values, y_values, f_slope_values, f_intercept_values, Gfitted_slope_values, Gfitted_intercept_values;
        
        for (const auto& point : dataset) {
            x_values.push_back(point.x);
            y_values.push_back(point.y);
            f_slope_values.push_back(fitted_slope);
            f_intercept_values.push_back(fitted_intercept);
            Gfitted_slope_values.push_back(Gfitted_slope);
            Gfitted_intercept_values.push_back(Gfitted_intercept);
        }

        // Plotting the original data points
        plt::scatter(x_values, y_values, 10.0); // Scatter plot for original data

        // Plotting the fitted line from normal linear regression
        std::vector<double> fitted_line;
        for (const auto& x : x_values) {
            fitted_line.push_back(fitted_slope * x + fitted_intercept);
        }
        plt::plot(x_values, fitted_line, "r-");
        plt::named_plot("Fitted Line", x_values, fitted_line); // Red line for normal linear regression

        // Plotting the fitted line from gradient descent
        std::vector<double> Gfitted_line;
        for (const auto& x : x_values) {
            Gfitted_line.push_back(Gfitted_slope * x + Gfitted_intercept);
        }
        plt::plot(x_values, Gfitted_line, "g-");
        plt::named_plot("Gradient Descent Line", x_values, Gfitted_line); // Green line for gradient descent

        plt::legend();
        plt::title("Linear Regression Fit");
        plt::xlabel("x");
        plt::ylabel("y");
        plt::show();

    }
};

#endif // PLOT_H
