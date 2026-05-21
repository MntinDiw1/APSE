#pragma once

#include <vector>

#include "sklearn_cpp/utils/metrics.hpp"
#include "sklearn_cpp/utils/visualize.hpp"

namespace sklearn_cpp {
namespace linear_model {

// Class to perform linear regression using gradient descent
class LinearRegression {
private:
    double learning_rate;
    int epochs;
    std::vector<double> w;
    double b;
public:
    // Construct with chosen learning rate and number of interations
    LinearRegression(double learning_rate = 1e-7, int epochs = 10000)
        : learning_rate(learning_rate), epochs(epochs), b(0.0) {}

    // Train the model on feature matrix X and label vector y using gradient descent
    void fit(const std::vector<std::vector<double>>& X, 
             const std::vector<double>& y) {
        int n = X.size();
        if (n == 0) return;
        int k = X[0].size();
        w.assign(k, 0.0);
        b = 0.0;

        for (int it = 0; it < epochs; ++it) {
            std::vector<double> dL_dw(k, 0.0);
            double dL_db = 0.0;

            for (int i = 0; i < n; ++i) {
                double prediction = b;
                for (int j = 0; j < k; ++j) {
                    prediction += w[j] * X[i][j];
                }
                double error = prediction - y[i];

                for (int j = 0; j < k; ++j) {
                    dL_dw[j] += error * X[i][j];
                }
                dL_db += error;
            }

            for (int j = 0; j < k; ++j) {
                w[j] -= (2.0 / n) * learning_rate * dL_dw[j];
            }
            b -= (2.0 / n) * learning_rate * dL_db;
        }
    }

    // Predict outputs for new samples
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const {
        int n = X.size();
        int k = w.size();
        std::vector<double> predictions(n);

        for (int i = 0; i < n; ++i) {
            double prediction = b;
            for (int j = 0; j < k; ++j) {
                prediction += w[j] * X[i][j];
            }
            predictions[i] = prediction;
        }
        return predictions;
    }

    // Return the coefficient of determination R squared
    double score(const std::vector<std::vector<double>>& X, 
                 const std::vector<double>& y_true) const {
        return utils::r2_score(y_true, predict(X));
    }

    // Accessors for the learned parameters
    const std::vector<double>& weights() const { return w; }
    double bias() const { return b; }

};

} // namespace linear_model
} // namespace sklearn_cpp