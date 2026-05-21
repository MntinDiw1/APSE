#pragma once


#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <vector>

#include "sklearn_cpp/utils/metrics.hpp"

namespace sklearn_cpp {
namespace linear_model {

    

class LogisticRegression {
public:
        // Construct a new LogisticRegression object
        LogisticRegression(double learning_rate = 0.1,
                                            int    n_iterations  = 1000,
                                            double lambda        = 0.01)
                : learning_rate_(learning_rate),
                  n_iterations_(n_iterations),
                  lambda_(lambda),
                  n_classes_(0),
                  bias_binary_(0.0) {}
    
        // Train the model.  Detects K from y and dispatches.
    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<double>&              y)
    {
        // Count distinct integer labels in y
        std::set<int> unique;   
        for (double v : y) unique.insert(static_cast<int>(v));
        n_classes_ = static_cast<int>(unique.size());

        if (n_classes_ < 2)
            throw std::invalid_argument(
                "LogisticRegression: need at least 2 classes in y");
 
        if (n_classes_ == 2)
            fit_binary_(X, y);
        else
            fit_multiclass_(X, y);
    }

     // Predict integer class labels for new samples.

    std::vector<int> predict(const std::vector<std::vector<double>>& X) const
    {
        if (n_classes_ == 2)
            return predict_binary_(X);
        return predict_multiclass_(X);
    }

     // Classification accuracy on (X, y_true). 
    double score(const std::vector<std::vector<double>>& X,
                 const std::vector<double>&              y_true) const
    {
        return utils::accuracy_score(y_true, predict(X)); 
    }

    int n_classes() const { return n_classes_; } 


    //Numerically stable sigmoid:   sigma(z) = 1 / (1 + exp(-z))
    double sigmoid(double z) const
    {
        // Guard against overflow for very negative z
        if (z >= 0.0) {
            const double ez = std::exp(-z);
            return 1.0 / (1.0 + ez);
        } else {
            const double ez = std::exp(z);
            return ez / (1.0 + ez);
        }
    }

     //Numerically stable softmax over a single sample's K raw scores.
     // Subtracts max(z) before exponentiation to avoid overflow.
    
    std::vector<double> softmax(const std::vector<double>& z) const
    {
        std::vector<double> out(z.size());

        // 1) find max for numerical stability
        double zmax = z[0];
        for (double v : z) if (v > zmax) zmax = v;

        // 2) exponentiate (z - zmax) and accumulate
        double sum = 0.0;
        for (std::size_t k = 0; k < z.size(); ++k) {
            out[k] = std::exp(z[k] - zmax);
            sum   += out[k];
        }

        // 3) normalise to a probability distribution
        for (double& v : out) v /= sum;
        return out;
    }


private:
    //  Hyper-parameters and learned parameters
    double learning_rate_;
    int    n_iterations_;
    double lambda_;
    int    n_classes_;        // set during fit()

    // Binary path  (K = 2): one weight vector and scalar bias
    std::vector<double> weights_binary_;   // shape [n_features]
    double              bias_binary_;

    // Multi-class path  (K > 2): one weight vector and bias per class
    std::vector<std::vector<double>> W_;   // shape [K][n_features]
    std::vector<double>              b_;   // shape [K]


    void fit_binary_(const std::vector<std::vector<double>>& X,
                     const std::vector<double>&              y)
    {
        
        const std::size_t m = X.size();
        const std::size_t n = X[0].size();
        weights_binary_.assign(n, 0.0);
        bias_binary_ = 0.0;

        for (int it = 0; it < n_iterations_; ++it) {

            std::vector<double> dw(n, 0.0);
            double              db = 0.0;

            for (std::size_t i = 0; i < m; ++i) {
                // forward pass: linear combination + sigmoid
                double z = bias_binary_;
                for (std::size_t j = 0; j < n; ++j)
                    z += weights_binary_[j] * X[i][j];
                const double y_hat = sigmoid(z);

                // accumulate gradient contributions
                const double err = y_hat - y[i];
                for (std::size_t j = 0; j < n; ++j)
                    dw[j] += err * X[i][j];
                db += err;
            }

            // gradient step with L2 regularisation on weights only
            const double scale = 1.0 / static_cast<double>(m);
            for (std::size_t j = 0; j < n; ++j) {
                const double grad = scale * dw[j] + 2.0 * lambda_ * weights_binary_[j];
                weights_binary_[j] -= learning_rate_ * grad;
            }
            bias_binary_ -= learning_rate_ * scale * db;
        }

        (void)X; (void)y;
    }

    /**
     *  For each row of X, compute sigmoid(w . x + b) and threshold at 0.5.
     */
    std::vector<int> predict_binary_(const std::vector<std::vector<double>>& X) const
    {
        std::vector<int> out(X.size(), 0);
        for (std::size_t i = 0; i < X.size(); ++i) {
            double z = bias_binary_;
            for (std::size_t j = 0; j < weights_binary_.size(); ++j)
                z += weights_binary_[j] * X[i][j];
            out[i] = (sigmoid(z) >= 0.5) ? 1 : 0;
        }
        return out;
    }


    
     // Train W_ and b_ using gradient descent on categorical cross-entropy with L2 regularisation.

    void fit_multiclass_(const std::vector<std::vector<double>>& X,
                         const std::vector<double>&              y)
    {
        const std::size_t m = X.size();
        const std::size_t n = X[0].size();
        const int         K = n_classes_;

        // Initialise weight matrix and bias vector to zeros
        W_.assign(K, std::vector<double>(n, 0.0));
        b_.assign(K, 0.0);

        for (int it = 0; it < n_iterations_; ++it) {

            // Accumulators for the gradients
            std::vector<std::vector<double>> dW(K, std::vector<double>(n, 0.0));
            std::vector<double>              db(K, 0.0);

            for (std::size_t i = 0; i < m; ++i) {

                // 1) forward pass: raw scores z_k = W_k . x_i + b_k
                std::vector<double> z(K, 0.0);
                for (int k = 0; k < K; ++k) {
                    double s = b_[k];
                    for (std::size_t j = 0; j < n; ++j) {
                        s += W_[k][j] * X[i][j];
                    }
                    z[k] = s;
                }

                // 2) squash raw scores into a probability distribution
                const std::vector<double> probs = softmax(z);

                // 3) compute errors and accumulate gradients
                // Performance tip applied: cast label once per sample, not per class
                const int label = static_cast<int>(y[i]); 
                
                for (int k = 0; k < K; ++k) {
                    // on-the-fly one-hot encoding comparison
                    const double err = probs[k] - ((k == label) ? 1.0 : 0.0);
                    
                    for (std::size_t j = 0; j < n; ++j) {
                        dW[k][j] += err * X[i][j];
                    }
                    db[k] += err;
                }
            }

            // 4) gradient update step with L2 regularisation
            const double scale = 1.0 / static_cast<double>(m);
            for (int k = 0; k < K; ++k) {
                for (std::size_t j = 0; j < n; ++j) {
                    const double grad = scale * dW[k][j] + 2.0 * lambda_ * W_[k][j];
                    W_[k][j] -= learning_rate_ * grad;
                }
                b_[k] -= learning_rate_ * scale * db[k];
            }
        }
    }

    /**
     * For each row of X, compute softmax(W x + b) and return argmax_k.
     */
    std::vector<int>
    predict_multiclass_(const std::vector<std::vector<double>>& X) const
    {
        std::vector<int> out(X.size(), 0);
        
        for (std::size_t i = 0; i < X.size(); ++i) {
            std::vector<double> z(n_classes_, 0.0);
            
            // 1) Compute raw scores for all K classes
            for (int k = 0; k < n_classes_; ++k) {
                double s = b_[k];
                for (std::size_t j = 0; j < W_[k].size(); ++j) {
                    s += W_[k][j] * X[i][j];
                }
                z[k] = s;
            }
            
            // 2) Get probabilities using the helper
            const std::vector<double> probs = softmax(z);
            
            // 3) Find the index of the highest probability (argmax)
            out[i] = static_cast<int>(
                std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()))
            );
        }
        
        return out;
    }
};

} // namespace linear_model
} // namespace sklearn_cpp
