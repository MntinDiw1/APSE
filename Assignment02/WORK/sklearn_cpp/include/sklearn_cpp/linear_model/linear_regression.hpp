#pragma once

/*
 * linear_regression.hpp
 * =====================
 * Mirrors sklearn.linear_model.LinearRegression from Python.
 *
 *     Model:    y_hat = w . x + b
 *     Loss:     MSE = (1/m) * sum_i (y_hat_i - y_i)^2
 *     Training: gradient descent on (w, b)
 *
 * Maintained by: Person A
 * ----------------------------------------------------------------------
 *  WHAT YOU NEED TO IMPLEMENT (search for "TODO" markers below):
 *
 *     1. fit(X, y)
 *        Initialise weights_ to zeros and bias_ to 0.
 *        Loop n_iterations_ times:
 *            - compute predictions y_hat for every sample
 *            - compute error_i = y_hat_i - y_i  for every sample
 *            - compute the gradients:
 *                   dL/dw_j = (2/m) * sum_i error_i * x_ij
 *                   dL/db   = (2/m) * sum_i error_i
 *            - update:
 *                   w_j -= learning_rate_ * dL/dw_j
 *                   b   -= learning_rate_ * dL/db
 *
 *     2. predict(X)
 *        Return a vector where every entry is  w . x + b  for that sample.
 *
 *     3. score(X, y_true)
 *        Call predict() and pass the result, together with y_true, to
 *        sklearn_cpp::utils::r2_score (already provided).
 *
 *  HYPERPARAMETER TIPS:
 *     The features in concrete.csv vary by 2-3 orders of magnitude, so
 *     gradient descent will diverge with the default lr.
 *     Either use a small lr (e.g. 1e-7) for many iterations,
 *     or normalise features inside fit() before training.
 * ----------------------------------------------------------------------
 */

#include <vector>

#include "sklearn_cpp/utils/metrics.hpp"

namespace sklearn_cpp {
namespace linear_model {

class LinearRegression {
public:
    /**
     * @param learning_rate  step size for gradient descent
     * @param n_iterations   number of gradient descent passes over the data
     */
    explicit LinearRegression(double learning_rate = 1e-7,
                              int    n_iterations  = 5000)
        : learning_rate_(learning_rate),
          n_iterations_(n_iterations),
          bias_(0.0) {}

    /**
     * Train the model on (X, y).
     *
     * @param X  feature matrix,  shape [n_samples][n_features]
     * @param y  target vector,   shape [n_samples]
     */
    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<double>&              y)
    {
        // ============================================================
        //  TODO  (Person A) -- implement gradient descent here.
        // ============================================================
        //  Suggested skeleton:
        //
        //  const std::size_t m = X.size();
        //  const std::size_t n = X[0].size();
        //  weights_.assign(n, 0.0);
        //  bias_ = 0.0;
        //
        //  for (int it = 0; it < n_iterations_; ++it) {
        //
        //      std::vector<double> dw(n, 0.0);
        //      double              db = 0.0;
        //
        //      // accumulate gradients across all samples
        //      for (std::size_t i = 0; i < m; ++i) {
        //          double y_hat = bias_;
        //          for (std::size_t j = 0; j < n; ++j)
        //              y_hat += weights_[j] * X[i][j];
        //
        //          const double err = y_hat - y[i];
        //          for (std::size_t j = 0; j < n; ++j)
        //              dw[j] += err * X[i][j];
        //          db += err;
        //      }
        //
        //      // gradient step (note factor 2/m from the MSE derivative)
        //      const double scale = 2.0 / static_cast<double>(m);
        //      for (std::size_t j = 0; j < n; ++j)
        //          weights_[j] -= learning_rate_ * scale * dw[j];
        //      bias_ -= learning_rate_ * scale * db;
        //  }

        (void)X; (void)y; // suppress unused-arg warnings until you implement
    }

    /**
     * Predict continuous outputs for new samples.
     */
    std::vector<double> predict(const std::vector<std::vector<double>>& X) const
    {
        // ============================================================
        //  TODO  (Person A) -- compute  w . x + b  for every sample.
        // ============================================================
        //
        //  std::vector<double> y_pred(X.size(), 0.0);
        //  for (std::size_t i = 0; i < X.size(); ++i) {
        //      double s = bias_;
        //      for (std::size_t j = 0; j < weights_.size(); ++j)
        //          s += weights_[j] * X[i][j];
        //      y_pred[i] = s;
        //  }
        //  return y_pred;

        return std::vector<double>(X.size(), 0.0); // placeholder
    }

    /**
     * Coefficient of determination (R^2).
     */
    double score(const std::vector<std::vector<double>>& X,
                 const std::vector<double>&              y_true) const
    {
        return utils::r2_score(y_true, predict(X));
    }

    // ------- Inspection helpers (useful for debugging) ----------------
    const std::vector<double>& weights() const { return weights_; }
    double                     bias()    const { return bias_;    }

private:
    double learning_rate_;
    int    n_iterations_;

    std::vector<double> weights_; // w, length = n_features
    double              bias_;    // b
};

} // namespace linear_model
} // namespace sklearn_cpp
