#pragma once
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <vector>
#include "sklearn_cpp/utils/metrics.hpp"

namespace sklearn_cpp {
namespace linear_model {

// LogisticRegression class
// Implements logistic regression using gradient descent.
// Supports both binary (2-class) and multiclass (K-class, K>2) classification.
class LogisticRegression {
private:
    // Hyperparameters (set by user or default)
    double learning_rate;
    int epochs;
    double lambda;

    // Set during training (automatically detected)
    int n_classes;

    // Parameters for Binary Classification (K = 2)
    std::vector<double> weights_binary;
    double bias_binary;

    // Parameters for Multiclass Classification (K > 2)
    std::vector<std::vector<double>> W;   // weights [K][n_features]
    std::vector<double> b;                // bias per class

public:
    // Constructor: initialize hyperparameters and parameters
    LogisticRegression(double learning_rate = 0.1, 
                       int epochs = 1000, 
                       double lambda = 0.01)
        : learning_rate(learning_rate),
          epochs(epochs),
          lambda(lambda),
          n_classes(0),
          bias_binary(0.0) {}

    // Train the model on features X and labels y
    // Automatically detects if the problem is binary or multiclass and dispatches accordingly
    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<double>& y)
    {
        // Count unique class labels in y
        std::set<int> unique;
        for (double v : y) 
            unique.insert(static_cast<int>(v));

        n_classes = static_cast<int>(unique.size());

        if (n_classes < 2)
            throw std::invalid_argument("LogisticRegression: need at least 2 classes in y");

        // Dispatch to the correct training function
        if (n_classes == 2)
            fit_binary(X, y);
        else
            fit_multiclass(X, y);
    }

    // Predict class labels for new data X
    std::vector<int> predict(const std::vector<std::vector<double>>& X) const
    {
        if (n_classes == 2)
            return predict_binary(X);
        else
            return predict_multiclass(X);
    }

    // Return model accuracy (fraction of correct predictions)
    double score(const std::vector<std::vector<double>>& X,
                 const std::vector<double>& y_true) const
    {
        return utils::accuracy_score(y_true, predict(X));
    }

    // Get the number of unique classes detected during training
    int get_n_classes() const { return n_classes; }

private:
    // Numerically stable sigmoid function
    // Maps any real value to (0, 1). Used for binary classification.
    double sigmoid(double z) const
    {
        if (z >= 0.0)
            return 1.0 / (1.0 + std::exp(-z));
        else
            return std::exp(z) / (1.0 + std::exp(z));
    }

    // Numerically stable softmax function
    // Converts a vector of raw scores to probabilities that sum to 1. Used for multiclass classification.
    std::vector<double> softmax(const std::vector<double>& z) const
    {
        std::vector<double> out(z.size());
        double zmax = z[0];
        for (double v : z) if (v > zmax) zmax = v;

        double sum = 0.0;
        for (int k = 0; k < (int)z.size(); ++k) {
            out[k] = std::exp(z[k] - zmax);
            sum += out[k];
        }
        for (double& v : out) v /= sum;
        return out;
    }

    // Train for Binary Classification (K = 2) using sigmoid and gradient descent
    void fit_binary(const std::vector<std::vector<double>>& X,
                    const std::vector<double>& y)
    {
        const int m = X.size();      // number of samples
        const int n = X[0].size();   // number of features

        // Initialize weights and bias to zero
        weights_binary.assign(n, 0.0);
        bias_binary = 0.0;

        for (int it = 0; it < epochs; ++it)
        {
            std::vector<double> dw(n, 0.0);
            double db = 0.0;

            for (int i = 0; i < m; ++i)
            {
                double z = bias_binary;
                for (int j = 0; j < n; ++j)
                    z += weights_binary[j] * X[i][j];

                double y_hat = sigmoid(z);
                double err = y_hat - y[i];

                for (int j = 0; j < n; ++j)
                    dw[j] += err * X[i][j];
                db += err;
            }

            const double scale = 1.0 / m;
            for (int j = 0; j < n; ++j)
            {
                double grad = scale * dw[j] + 2.0 * lambda * weights_binary[j];
                weights_binary[j] -= learning_rate * grad;
            }
            bias_binary -= learning_rate * scale * db;
        }
    }

    // Train for Multiclass Classification (K > 2) using softmax and gradient descent
    void fit_multiclass(const std::vector<std::vector<double>>& X,
                        const std::vector<double>& y)
    {
        const int m = X.size();
        const int n = X[0].size();
        const int K = n_classes;

        // Initialize weights and biases for each class to zero
        W.assign(K, std::vector<double>(n, 0.0));
        b.assign(K, 0.0);

        for (int it = 0; it < epochs; ++it)
        {
            std::vector<std::vector<double>> dW(K, std::vector<double>(n, 0.0));
            std::vector<double> db(K, 0.0);

            for (int i = 0; i < m; ++i)
            {
                std::vector<double> z(K, 0.0);
                for (int k = 0; k < K; ++k)
                {
                    double s = b[k];
                    for (int j = 0; j < n; ++j)
                        s += W[k][j] * X[i][j];
                    z[k] = s;
                }

                const std::vector<double> probs = softmax(z);
                const int label = static_cast<int>(y[i]);

                for (int k = 0; k < K; ++k)
                {
                    const double err = probs[k] - ((k == label) ? 1.0 : 0.0);
                    for (int j = 0; j < n; ++j)
                        dW[k][j] += err * X[i][j];
                    db[k] += err;
                }
            }

            const double scale = 1.0 / m;
            for (int k = 0; k < K; ++k)
            {
                for (int j = 0; j < n; ++j)
                {
                    const double grad = scale * dW[k][j] + 2.0 * lambda * W[k][j];
                    W[k][j] -= learning_rate * grad;
                }
                b[k] -= learning_rate * scale * db[k];
            }
        }
    }

    // Predict class labels for binary classification (K = 2)
    std::vector<int> predict_binary(const std::vector<std::vector<double>>& X) const
    {
        std::vector<int> out(X.size(), 0);
        for (int i = 0; i < (int)X.size(); ++i)
        {
            double z = bias_binary;
            for (int j = 0; j < (int)weights_binary.size(); ++j)
                z += weights_binary[j] * X[i][j];

            out[i] = (sigmoid(z) >= 0.5) ? 1 : 0;
        }
        return out;
    }

    // Predict class labels for multiclass classification (K > 2)
    std::vector<int> predict_multiclass(const std::vector<std::vector<double>>& X) const
    {
        std::vector<int> out(X.size(), 0);
        for (int i = 0; i < (int)X.size(); ++i)
        {
            std::vector<double> z(n_classes, 0.0);

            for (int k = 0; k < n_classes; ++k)
            {
                double s = b[k];
                for (int j = 0; j < (int)W[k].size(); ++j)
                    s += W[k][j] * X[i][j];
                z[k] = s;
            }

            const std::vector<double> probs = softmax(z);

            out[i] = static_cast<int>(
                std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()))
            );
        }
        return out;
    }
};

} // namespace linear_model
} // namespace sklearn_cpp