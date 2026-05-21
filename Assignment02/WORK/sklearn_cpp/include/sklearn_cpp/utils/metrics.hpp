#pragma once

/*
 * metrics.hpp
 * ===========
 * Shared evaluation metrics for sklearn_cpp.  Header-only.
 *
 * Provides:
 *     r2_score(y_true, y_pred)        coefficient of determination
 *     accuracy_score(y_true, y_pred)  classification accuracy
 *
 * Used by all three main programs (concrete, ecg, mnist).
 */

#include <stdexcept>
#include <vector>

namespace sklearn_cpp {
namespace utils {

/**
 * R^2  (coefficient of determination).
 *
 *     R^2 = 1 - SS_res / SS_tot
 *
 *     SS_res = sum_i (y_true_i - y_pred_i)^2
 *     SS_tot = sum_i (y_true_i - y_mean)^2
 *
 * R^2 close to 1.0  -> excellent fit
 * R^2 close to 0    -> not better than predicting the mean
 * R^2 negative      -> model is worse than predicting the mean
 *
 * @throws std::invalid_argument if the two vectors have different sizes.
 */
inline double r2_score(const std::vector<double>& y_true,
                       const std::vector<double>& y_pred)
{
    if (y_true.size() != y_pred.size())
        throw std::invalid_argument("r2_score: vector size mismatch");
    if (y_true.empty())
        throw std::invalid_argument("r2_score: empty input vectors");

    // Mean of y_true
    double sum = 0.0;
    for (double v : y_true) sum += v;
    const double mean = sum / static_cast<double>(y_true.size());

    // Residual and total sums of squares
    double ss_res = 0.0;
    double ss_tot = 0.0;
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        const double d_res = y_true[i] - y_pred[i];
        const double d_tot = y_true[i] - mean;
        ss_res += d_res * d_res;
        ss_tot += d_tot * d_tot;
    }

    // Degenerate case: every target is the same -> tot=0
    if (ss_tot == 0.0) return (ss_res == 0.0) ? 1.0 : 0.0;

    return 1.0 - ss_res / ss_tot;
}

/**
 * Classification accuracy: fraction of predictions that match the truth.
 *
 *     accuracy = (# correct) / (# total)
 *
 * Returns a value in [0, 1].
 */
inline double accuracy_score(const std::vector<int>&    y_true,
                             const std::vector<int>&    y_pred)
{
    if (y_true.size() != y_pred.size())
        throw std::invalid_argument("accuracy_score: vector size mismatch");
    if (y_true.empty())
        throw std::invalid_argument("accuracy_score: empty input vectors");

    std::size_t correct = 0;
    for (std::size_t i = 0; i < y_true.size(); ++i)
        if (y_true[i] == y_pred[i]) ++correct;

    return static_cast<double>(correct) / static_cast<double>(y_true.size());
}

/**
 * Overload of accuracy_score for when y_true comes from a CSV as double
 * (labels like 0.0 and 1.0).  Internally casts to int.
 */
inline double accuracy_score(const std::vector<double>& y_true,
                             const std::vector<int>&    y_pred)
{
    if (y_true.size() != y_pred.size())
        throw std::invalid_argument("accuracy_score: vector size mismatch");

    std::vector<int> y_int(y_true.size());
    for (std::size_t i = 0; i < y_true.size(); ++i)
        y_int[i] = static_cast<int>(y_true[i]);

    return accuracy_score(y_int, y_pred);
}

}
} 
