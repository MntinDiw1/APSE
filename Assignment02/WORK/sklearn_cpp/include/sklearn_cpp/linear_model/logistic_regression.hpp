#pragma once

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <vector>

#include "sklearn_cpp/utils/metrics.hpp"

namespace sklearn_cpp {
namespace linear_model {

// Class to perform logistic regression for both binary and multi-class problems
class LogisticRegression {
public:
    // Construct with chosen learning rate, epochs, and L2 regularisation strength
    LogisticRegression(double learning_rate = 0.1, int epochs = 1000, double lambda = 0.01)
        : learning_rate(learning_rate), epochs(epochs), lambda(lambda),
          n_classes(0), b_bin(0.0) {}

    // Train the model - detects number of classes from y and runs the appropriate path
    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<double>& y) {
        std::set<int> unique;
        for (double v : y) unique.insert(static_cast<int>(v));
        n_classes = static_cast<int>(unique.size());

        if (n_classes < 2)
            throw std::invalid_argument("LogisticRegression needs at least 2 classes in y");

        if (n_classes == 2) fit_binary(X, y);
        else                fit_multiclass(X, y);
    }

    // Predict integer class labels for new samples
    std::vector<int> predict(const std::vector<std::vector<double>>& X) const {
        if (n_classes == 2) return predict_binary(X);
        return predict_multiclass(X);
    }

    // Return classification accuracy on the training set
    double score(const std::vector<std::vector<double>>& X,
                 const std::vector<double>& y_true) const {
        return utils::accuracy_score(y_true, predict(X));
    }

    // Accessor for the number of classes detected during fit
    int num_classes() const { return n_classes; }

private:
    double learning_rate;
    int epochs;
    double lambda;
    int n_classes;

    // Binary path: one weight vector and one scalar bias
    std::vector<double> w_bin;
    double b_bin;

    // Multi-class path: one weight vector per class and one bias per class
    std::vector<std::vector<double>> W;
    std::vector<double> B;

    // Numerically stable sigmoid that avoids overflow for large negative z
    static double sigmoid(double z) {
        if (z >= 0.0) {
            return 1.0 / (1.0 + std::exp(-z));
        } else {
            double ez = std::exp(z);
            return ez / (1.0 + ez);
        }
    }

    // Numerically stable softmax that subtracts the max before exponentiating
    static std::vector<double> softmax(const std::vector<double>& z) {
        std::vector<double> probs(z.size());
        double zmax = z[0];
        for (double v : z) if (v > zmax) zmax = v;

        double sum = 0.0;
        int K = z.size();
        for (int c = 0; c < K; ++c) {
            probs[c] = std::exp(z[c] - zmax);
            sum += probs[c];
        }
        for (double& v : probs) v /= sum;
        return probs;
    }

    // Train w_bin and b_bin using gradient descent on binary cross entropy with L2 regularisation
    void fit_binary(const std::vector<std::vector<double>>& X,
                    const std::vector<double>& y) {
        int n = X.size();
        if (n == 0) return;
        int k = X[0].size();
        w_bin.assign(k, 0.0);
        b_bin = 0.0;

        for (int it = 0; it < epochs; ++it) {
            std::vector<double> dL_dw(k, 0.0);
            double dL_db = 0.0;

            for (int i = 0; i < n; ++i) {
                double z = b_bin;
                for (int j = 0; j < k; ++j) {
                    z += w_bin[j] * X[i][j];
                }
                double y_hat = sigmoid(z);
                double error = y_hat - y[i];

                for (int j = 0; j < k; ++j) {
                    dL_dw[j] += error * X[i][j];
                }
                dL_db += error;
            }

            // L2 regularisation applies to weights only, not the bias
            for (int j = 0; j < k; ++j) {
                double grad = (1.0 / n) * dL_dw[j] + 2.0 * lambda * w_bin[j];
                w_bin[j] -= learning_rate * grad;
            }
            b_bin -= learning_rate * (1.0 / n) * dL_db;
        }
    }

    // Predict 0 or 1 for each sample by thresholding sigmoid output at 0.5
    std::vector<int> predict_binary(const std::vector<std::vector<double>>& X) const {
        int n = X.size();
        int k = w_bin.size();
        std::vector<int> predictions(n);

        for (int i = 0; i < n; ++i) {
            double z = b_bin;
            for (int j = 0; j < k; ++j) {
                z += w_bin[j] * X[i][j];
            }
            predictions[i] = (sigmoid(z) >= 0.5) ? 1 : 0;
        }
        return predictions;
    }

    // Train W and B using softmax gradient descent with L2 regularisation
    void fit_multiclass(const std::vector<std::vector<double>>& X,
                        const std::vector<double>& y) {
        int n = X.size();
        if (n == 0) return;
        int k = X[0].size();
        int K = n_classes;

        W.assign(K, std::vector<double>(k, 0.0));
        B.assign(K, 0.0);

        for (int it = 0; it < epochs; ++it) {
            std::vector<std::vector<double>> dL_dW(K, std::vector<double>(k, 0.0));
            std::vector<double> dL_dB(K, 0.0);

            for (int i = 0; i < n; ++i) {
                // Compute the raw score for each class
                std::vector<double> z(K, 0.0);
                for (int c = 0; c < K; ++c) {
                    double s = B[c];
                    for (int j = 0; j < k; ++j) {
                        s += W[c][j] * X[i][j];
                    }
                    z[c] = s;
                }

                // Convert scores into probabilities via softmax
                std::vector<double> probs = softmax(z);

                // Error per class is predicted probability minus the one-hot target
                int label = static_cast<int>(y[i]);
                for (int c = 0; c < K; ++c) {
                    double error = probs[c] - ((c == label) ? 1.0 : 0.0);
                    for (int j = 0; j < k; ++j) {
                        dL_dW[c][j] += error * X[i][j];
                    }
                    dL_dB[c] += error;
                }
            }

            // L2 regularisation applies to weights only, not the biases
            for (int c = 0; c < K; ++c) {
                for (int j = 0; j < k; ++j) {
                    double grad = (1.0 / n) * dL_dW[c][j] + 2.0 * lambda * W[c][j];
                    W[c][j] -= learning_rate * grad;
                }
                B[c] -= learning_rate * (1.0 / n) * dL_dB[c];
            }
        }
    }

    // For each sample compute softmax probabilities and return the class with the highest one
    std::vector<int> predict_multiclass(const std::vector<std::vector<double>>& X) const {
        int n = X.size();
        int k = W[0].size();
        int K = n_classes;
        std::vector<int> predictions(n);

        for (int i = 0; i < n; ++i) {
            std::vector<double> z(K, 0.0);
            for (int c = 0; c < K; ++c) {
                double s = B[c];
                for (int j = 0; j < k; ++j) {
                    s += W[c][j] * X[i][j];
                }
                z[c] = s;
            }
            std::vector<double> probs = softmax(z);

            // Find the class with the highest probability
            int best = 0;
            for (int c = 1; c < K; ++c) {
                if (probs[c] > probs[best]) best = c;
            }
            predictions[i] = best;
        }
        return predictions;
    }
};

} // namespace linear_model
} // namespace sklearn_cpp