#pragma once

#include <vector>

#include "sklearn_cpp/utils/metrics.hpp"


// Namespacing to avoid potential conflicts with other libraries
namespace sklearn_cpp {
namespace linear_model {

// Class to perform linear regression using gradient descent.
// Mirrors the interface of sklearn.linear_model.LinearRegression in Python.
class LinearRegression {
private:
    // Hyperparameters set by the user via the constructor
    double learning_rate;
    int epochs;

    // Learned parameters - populated during fit()
    std::vector<double> w;  // weight vector, one entry per feature
    double b;               // bias (intercept)

public:
    // Construct with chosen learning rate and number of iterations.
    // Default values work well for the concrete dataset.
    LinearRegression(double learning_rate = 1e-7, int epochs = 10000)
        : learning_rate(learning_rate), epochs(epochs), b(0.0) {}

    // Train the model on feature matrix X and label vector y using gradient descent.
    // Minimises the mean squared error loss to find the best w and b.
    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<double>& y) {
        int n = X.size();           // number of samples
        if (n == 0) return;
        int k = X[0].size();        // number of features per sample

        // Start fresh every time fit() is called
        w.assign(k, 0.0);
        b = 0.0;

        // Outer loop - one full pass over the data per iteration
        for (int it = 0; it < epochs; ++it) {

            // Accumulators for the gradient over all samples
            std::vector<double> dL_dw(k, 0.0);
            double dL_db = 0.0;

            // Compute the gradient contribution of each sample
            for (int i = 0; i < n; ++i) {
                // Forward pass: predicted value y_hat = w . x + b
                double prediction = b;
                for (int j = 0; j < k; ++j) {
                    prediction += w[j] * X[i][j];
                }

                // Error between predicted and actual value
                double error = prediction - y[i];

                // Accumulate partial derivatives of the loss
                for (int j = 0; j < k; ++j) {
                    dL_dw[j] += error * X[i][j];
                }
                dL_db += error;
            }

            // Update each weight and the bias by stepping in the
            // direction that reduces the loss most steeply.
            // The factor 2/n comes from differentiating the MSE.
            for (int j = 0; j < k; ++j) {
                w[j] -= (2.0 / n) * learning_rate * dL_dw[j];
            }
            b -= (2.0 / n) * learning_rate * dL_db;
        }
    }

    // Predict outputs for new samples using the learned weights and bias
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const {
        int n = X.size();
        int k = w.size();
        std::vector<double> predictions(n);

        // For each sample compute y_hat = w . x + b
        for (int i = 0; i < n; ++i) {
            double prediction = b;
            for (int j = 0; j < k; ++j) {
                prediction += w[j] * X[i][j];
            }
            predictions[i] = prediction;
        }
        return predictions;
    }

    // Return the R squared score on the given data (1.0 is a perfect fit)
    double score(const std::vector<std::vector<double>>& X,
                 const std::vector<double>& y_true) const {
        return utils::r2_score(y_true, predict(X));
    }

    // Accessors for inspecting the learned parameters after training
    const std::vector<double>& weights() const { return w; }
    double bias() const { return b; }
};

} // namespace linear_model
} // namespace sklearn_cpp