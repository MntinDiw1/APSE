#ifndef PLOT_H
#define PLOT_H
#define WITHOUT_NUMPY 

#include <vector>
#include <iostream>
#include <numeric> 
#include "coordinate.h"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp; 

class Plot { 
public:
    // Plotting function for 1D linear regression results
    static void plotData1D(const std::vector<coordinate>& dataset, const std::vector<double>& w_ols, double b_ols, const std::vector<double>& w_gd, double b_gd) {
        std::vector<double> x_values, y_values;
        double min_x = 1e9, max_x = -1e9;

        // Extract x and y values from the dataset and find min/max for plotting
        for (const auto& point : dataset) {
            if (point.x.empty()) continue;
            
            double x = point.x[0];
            x_values.push_back(x);
            y_values.push_back(point.y);

            if (x < min_x) min_x = x;
            if (x > max_x) max_x = x;
        }

        // Create points for the regression lines based on the min and max x values
        std::vector<double> line_x = {min_x, max_x};
        std::vector<double> line_y_ols = {
            w_ols[0] * min_x + b_ols,
            w_ols[0] * max_x + b_ols
        };
        std::vector<double> line_y_gd = {
            w_gd[0] * min_x + b_gd,
            w_gd[0] * max_x + b_gd
        };
        // Plotting the data points and regression lines
        plt::figure_size(800, 600);
        // Plot actual data points in blue, OLS line in red, and GD line in green
        plt::named_plot("Actual Data", x_values, y_values, "b.");
        plt::named_plot("OLS Line", line_x, line_y_ols, "r-");
        plt::named_plot("GD Line", line_x, line_y_gd, "g-");
        // Add legend, title, and axis labels
        plt::legend();
        plt::title("1D Linear Regression Fit");
        plt::xlabel("x");
        plt::ylabel("y");
        // Adjust layout, save the plot, and display it
        plt::tight_layout();
        plt::save("linear_regression_1d_plot.png");
        plt::show(); 
    }
    // Plotting function for 2D linear regression results
    static void plotData(const std::vector<coordinate>& dataset, const std::vector<double>& w_ols, double b_ols, const std::vector<double>& w_gd, double b_gd) {
        std::vector<double> x1_values, x2_values, y_values;
        // Extract x1, x2, and y values from the dataset and find min/max for plotting
        double sum_x1 = 0, sum_x2 = 0;
        double min_x1 = 1e9, max_x1 = -1e9;
        double min_x2 = 1e9, max_x2 = -1e9;
        // Loop through the dataset to populate x1_values, x2_values, and y_values, while also calculating sums and min/max for x1 and x2
        for (const auto& point : dataset) {
            if (point.x.size() < 2) continue;

            double x1 = point.x[0];
            double x2 = point.x[1];
            
            x1_values.push_back(x1);
            x2_values.push_back(x2);
            y_values.push_back(point.y);

            sum_x1 += x1;
            sum_x2 += x2;

            if (x1 < min_x1) min_x1 = x1;
            if (x1 > max_x1) max_x1 = x1;
            if (x2 < min_x2) min_x2 = x2;
            if (x2 > max_x2) max_x2 = x2;
        }
        // Calculate mean values for x1 and x2 to use in plotting the regression lines
        double mean_x1 = sum_x1 / dataset.size();
        double mean_x2 = sum_x2 / dataset.size();
        // Create points for the regression lines based on the min and max x1 and x2 values, while keeping the other variable at its mean to visualize the effect of each variable separately
        std::vector<double> line_x1 = {min_x1, max_x1};
        std::vector<double> line1_y_ols = {
            w_ols[0] * min_x1 + w_ols[1] * mean_x2 + b_ols,
            w_ols[0] * max_x1 + w_ols[1] * mean_x2 + b_ols
        };
        std::vector<double> line1_y_gd = {
            w_gd[0] * min_x1 + w_gd[1] * mean_x2 + b_gd,
            w_gd[0] * max_x1 + w_gd[1] * mean_x2 + b_gd
        };
        // For the second plot, we vary x2 while keeping x1 at its mean to visualize the effect of x2 on y
        std::vector<double> line_x2 = {min_x2, max_x2};
        std::vector<double> line2_y_ols = {
            w_ols[0] * mean_x1 + w_ols[1] * min_x2 + b_ols,
            w_ols[0] * mean_x1 + w_ols[1] * max_x2 + b_ols
        };
        std::vector<double> line2_y_gd = {
            w_gd[0] * mean_x1 + w_gd[1] * min_x2 + b_gd,
            w_gd[0] * mean_x1 + w_gd[1] * max_x2 + b_gd
        };
        // Plotting the data points and regression lines
        plt::figure_size(1200, 600);
        // First subplot for y vs x1, keeping x2 at its mean
        plt::subplot2grid(1, 2, 0, 0); 
        plt::named_plot("Actual Data", x1_values, y_values, "b.");
        plt::named_plot("OLS Line (x2 at mean)", line_x1, line1_y_ols, "r-");
        plt::named_plot("GD Line (x2 at mean)", line_x1, line1_y_gd, "g-");
        plt::legend();
        plt::title("Regression Fit (y vs x1)");
        plt::xlabel("x1");
        plt::ylabel("y");
        // Second subplot for y vs x2, keeping x1 at its mean
        plt::subplot2grid(1, 2, 0, 1); 
        plt::named_plot("Actual Data", x2_values, y_values, "b.");
        plt::named_plot("OLS Line (x1 at mean)", line_x2, line2_y_ols, "r-");
        plt::named_plot("GD Line (x1 at mean)", line_x2, line2_y_gd, "g-");
        plt::legend();
        plt::title("Regression Fit (y vs x2)");
        plt::xlabel("x2");
        plt::ylabel("y");
        // Adjust layout, save the plot, and display it
        plt::tight_layout();
        plt::save("linear_regression_plots.png");
        plt::show();
    }
};

#endif // PLOT_H