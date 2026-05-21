#pragma once

/*
 * visualize.hpp
 * Plotting utilities for sklearn_cpp. Header-only.
 *
 * Wraps matplotlibcpp to provide ready-made charts for machine learning 
 * evaluation, heavily utilizing modern C++ (C++17) standard library features 
 * for numerical safety and performance.
 *
 * Dependencies:
 *   Python 3 with matplotlib must be installed.
 *   Compile with: g++ ... $(python3-config --cflags --ldflags) -lmatplotlib_cpp
 */

#define WITHOUT_NUMPY
#include "matplotlibcpp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace sklearn_cpp {
namespace utils {

// Type aliases for readability and maintainability
using Matrix      = std::vector<std::vector<double>>;
using Vector      = std::vector<double>;
using LabelVector = std::vector<int>;

// Isolate internal helpers and aliases to prevent header namespace pollution
namespace detail {
    namespace plt = matplotlibcpp;

    /**
     * @brief Saves the figure if a path is provided, then displays it.
     */
    inline void save_and_show(const std::string& save_path) {
        if (!save_path.empty()) {
            plt::save(save_path);
        }
        plt::show();
    }
} // namespace detail


class Visualizer {
public:
    // ==================================================================
    //  LINEAR REGRESSION PLOTS
    // ==================================================================

    // Plots the raw 1-D feature data against the fitted regression line.
    static void plot_regression_fit(
            const Matrix& X,
            const Vector& y_true,
            const Vector& y_pred,
            const std::string& title = "Regression Fit",
            const std::string& save_path = "")
    {
        if (X.empty() || X.front().size() != 1) {
            throw std::invalid_argument("plot_regression_fit: X must have exactly one feature column.");
        }

        const std::size_t n = X.size();

        // Extract the single feature for the x-axis using standard transform
        Vector x_vals(n);
        std::transform(X.begin(), X.end(), x_vals.begin(), [](const Vector& row) { return row[0]; });

        // Generate indices and sort them based on x_vals to draw a continuous fit line
        std::vector<std::size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&x_vals](std::size_t a, std::size_t b) { 
            return x_vals[a] < x_vals[b]; 
        });

        // Reorder X and Y_pred based on the sorted indices
        Vector x_sorted(n), y_pred_sorted(n);
        for (std::size_t i = 0; i < n; ++i) {
            x_sorted[i]      = x_vals[idx[i]];
            y_pred_sorted[i] = y_pred[idx[i]];
        }

        detail::plt::figure_size(800, 600);
        detail::plt::named_plot("Data", x_vals, y_true, "b.");
        detail::plt::named_plot("Fit", x_sorted, y_pred_sorted, "r-");
        
        detail::plt::legend();
        detail::plt::title(title);
        detail::plt::xlabel("x");
        detail::plt::ylabel("y");
        detail::plt::tight_layout();
        detail::save_and_show(save_path);
    }

    // Plots actual vs predicted values. Points close to y=x imply high accuracy.
    static void plot_actual_vs_predicted(
            const Vector& y_true,
            const Vector& y_pred,
            const std::string& title = "Actual vs Predicted",
            const std::string& save_path = "")
    {
        if (y_true.size() != y_pred.size()) {
            throw std::invalid_argument("plot_actual_vs_predicted: Vector size mismatch.");
        }

        // Determine boundaries for the y=x perfect-prediction diagonal
        const auto [min_it, max_it] = std::minmax_element(y_true.begin(), y_true.end());
        const Vector diag = { *min_it, *max_it };

        detail::plt::figure_size(700, 700);
        detail::plt::named_plot("Predictions", y_true, y_pred, "b.");
        detail::plt::named_plot("Perfect fit", diag, diag, "r--");
        
        detail::plt::legend();
        detail::plt::title(title);
        detail::plt::xlabel("Actual Target");
        detail::plt::ylabel("Predicted Value");
        detail::plt::tight_layout();
        detail::save_and_show(save_path);
    }

    // Plots residuals (y_true - y_pred) to visually inspect homoscedasticity.
    static void plot_residuals(
            const Vector& y_true,
            const Vector& y_pred,
            const std::string& title = "Residual Plot",
            const std::string& save_path = "")
    {
        if (y_true.size() != y_pred.size()) {
            throw std::invalid_argument("plot_residuals: Vector size mismatch.");
        }

        const std::size_t n = y_true.size();
        Vector residuals(n);
        std::transform(y_true.begin(), y_true.end(), y_pred.begin(), residuals.begin(), std::minus<double>());

        // Zero reference line spanning the predicted range
        const auto [min_it, max_it] = std::minmax_element(y_pred.begin(), y_pred.end());
        const Vector zero_x = { *min_it, *max_it };
        const Vector zero_y = { 0.0, 0.0 };

        detail::plt::figure_size(800, 500);
        detail::plt::named_plot("Residuals", y_pred, residuals, "b.");
        detail::plt::named_plot("Zero line", zero_x, zero_y, "r--");
        
        detail::plt::legend();
        detail::plt::title(title);
        detail::plt::xlabel("Predicted value");
        detail::plt::ylabel("Residual (Actual - Predicted)");
        detail::plt::tight_layout();
        detail::save_and_show(save_path);
    }

    // ==================================================================
    //  LOGISTIC REGRESSION PLOTS
    // ==================================================================

    // Evaluates a classifier over a 2D mesh grid and plots the decision regions.
    // Did not implement this in the example, but it's a common and useful visualization for 2D datasets
    // I develouped this thinking binary meant 2D.
    static void plot_decision_boundary_2d( 
            const Matrix& X,
            const Vector& y_true,
            const std::function<LabelVector(const Matrix&)>& predictor,
            const std::string& title = "Decision Boundary",
            int resolution = 200,
            const std::string& save_path = "")
    {
        if (X.empty() || X.front().size() != 2) {
            throw std::invalid_argument("plot_decision_boundary_2d: X must be exactly 2-Dimensional.");
        }

        const std::size_t n = X.size();

        // --- Compute axis bounds with a 10% protective margin ---
        double x0_min = X[0][0], x0_max = X[0][0];
        double x1_min = X[0][1], x1_max = X[0][1];
        
        for (std::size_t i = 1; i < n; ++i) {
            x0_min = std::min(x0_min, X[i][0]);
            x0_max = std::max(x0_max, X[i][0]);
            x1_min = std::min(x1_min, X[i][1]);
            x1_max = std::max(x1_max, X[i][1]);
        }
        
        const double margin_0 = 0.1 * (x0_max - x0_min + 1e-9);
        const double margin_1 = 0.1 * (x1_max - x1_min + 1e-9);
        x0_min -= margin_0; x0_max += margin_0;
        x1_min -= margin_1; x1_max += margin_1;

        // --- Generate Evaluation Grid ---
        const double step0 = (x0_max - x0_min) / resolution;
        const double step1 = (x1_max - x1_min) / resolution;

        Matrix grid;
        grid.reserve(static_cast<std::size_t>(resolution * resolution));

        for (int r = 0; r < resolution; ++r) {
            for (int c = 0; c < resolution; ++c) {
                grid.push_back({x0_min + c * step0, x1_min + r * step1});
            }
        }

        // Evaluate model over the entire grid
        const LabelVector grid_labels = predictor(grid);

        // Map distinct classes discovered in targets or predictions
        std::set<int> class_set;
        for (double v : y_true) class_set.insert(static_cast<int>(v));
        for (int v : grid_labels) class_set.insert(v);

        const std::vector<int> classes(class_set.begin(), class_set.end());
        const std::size_t K = classes.size();

        // High-contrast qualitative color palettes
        constexpr const char* bg_colors[]  = {"#aec6e8", "#ffb347", "#90ee90", "#dda0dd", "#f08080", "#87ceeb"};
        constexpr const char* dot_colors[] = {"#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728", "#17becf"};
        constexpr const char* markers[]    = {"o", "s", "^", "D", "v", "P"};
        constexpr std::size_t palette_size = 6;

        detail::plt::figure_size(800, 700);

        // --- Render Background Decision Regions ---
        for (std::size_t ki = 0; ki < K; ++ki) {
            const int cls = classes[ki];
            Vector gx, gy;
            gx.reserve(grid.size());
            gy.reserve(grid.size());

            for (std::size_t gi = 0; gi < grid.size(); ++gi) {
                if (grid_labels[gi] == cls) {
                    gx.push_back(grid[gi][0]);
                    gy.push_back(grid[gi][1]);
                }
            }
            if (!gx.empty()) {
                const std::string color = bg_colors[ki % palette_size];
                detail::plt::scatter(gx, gy, 1.0, {{"color", color}, {"alpha", "0.4"}});
            }
        }

        // --- Render Ground-Truth Scatter Overlays ---
        for (std::size_t ki = 0; ki < K; ++ki) {
            const int cls = classes[ki];
            Vector px, py;
            px.reserve(n);
            py.reserve(n);

            for (std::size_t i = 0; i < n; ++i) {
                if (static_cast<int>(y_true[i]) == cls) {
                    px.push_back(X[i][0]);
                    py.push_back(X[i][1]);
                }
            }
            if (!px.empty()) {
                const std::string color  = dot_colors[ki % palette_size];
                const std::string marker = markers[ki % palette_size];
                detail::plt::scatter(px, py, 30.0, {
                    {"color", color},
                    {"edgecolors", "black"},
                    {"linewidths", "0.5"},
                    {"zorder", "3"},
                    {"label", "Class " + std::to_string(cls)}
                });
            }
        }

        detail::plt::legend();
        detail::plt::title(title);
        detail::plt::xlabel("Feature 0");
        detail::plt::ylabel("Feature 1");
        detail::plt::tight_layout();
        detail::save_and_show(save_path);
    }

    // Renders a color-coded confusion matrix heatmap.
    static void plot_confusion_matrix(
            const Vector& y_true,
            const LabelVector& y_pred,
            const std::string& title = "Confusion Matrix",
            const std::string& save_path = "")
    {
        if (y_true.size() != y_pred.size()) {
            throw std::invalid_argument("plot_confusion_matrix: Vector size mismatch.");
        }

        // Extract sorted unique classes
        std::set<int> cls_set;
        for (double v : y_true) cls_set.insert(static_cast<int>(v));
        for (int v : y_pred) cls_set.insert(v);
        
        const std::vector<int> classes(cls_set.begin(), cls_set.end());
        const int K = static_cast<int>(classes.size());

        // Map actual class integer to zero-indexed matrix position
        std::map<int, int> cls_idx;
        for (int ki = 0; ki < K; ++ki) {
            cls_idx[classes[ki]] = ki;
        }

        // Accumulate Confusion Matrix counts
        std::vector<std::vector<int>> cm(K, std::vector<int>(K, 0));
        for (std::size_t i = 0; i < y_true.size(); ++i) {
            const int ti = cls_idx[static_cast<int>(y_true[i])];
            const int pi = cls_idx[y_pred[i]];
            ++cm[ti][pi];
        }

        // Find the global maximum for color scale normalization
        int max_count = 0;
        for (int r = 0; r < K; ++r) {
            for (int c = 0; c < K; ++c) {
                max_count = std::max(max_count, cm[r][c]);
            }
        }

        // Fast, type-safe color interpolator mapping directly to Hex representation
        auto count_to_color = [&](int count) -> std::string {
            const double shade = (max_count > 0) ? static_cast<double>(count) / max_count : 0.0;
            
            // Linear interpolation to a Blues colormap; clamped to prevent overflow
            const int r = std::clamp(static_cast<int>(255.0 + shade * (49.0 - 255.0)), 0, 255);
            const int g = std::clamp(static_cast<int>(255.0 + shade * (130.0 - 255.0)), 0, 255);
            const int b = std::clamp(static_cast<int>(255.0 + shade * (189.0 - 255.0)), 0, 255);
            
            char hex_buf[8];
            std::snprintf(hex_buf, sizeof(hex_buf), "#%02x%02x%02x", r, g, b);
            return std::string(hex_buf);
        };

        // Render Heatmap cells via scatter markers to simulate imshow
        detail::plt::figure_size(100 + K * 80, 100 + K * 80);
        
        for (int r = 0; r < K; ++r) {
            for (int c = 0; c < K; ++c) {
                detail::plt::scatter(
                    Vector{static_cast<double>(c)},
                    Vector{static_cast<double>(r)},
                    1200.0,
                    {
                        {"color", count_to_color(cm[r][c])},
                        {"marker", "s"},
                        {"edgecolors", "none"}
                    }
                );
                
                // Overlay text count
                detail::plt::text(static_cast<double>(c), static_cast<double>(r), std::to_string(cm[r][c]));
            }
        }

        detail::plt::xlim(-0.5, static_cast<double>(K) - 0.5);
        detail::plt::ylim(static_cast<double>(K) - 0.5, -0.5);

        // Configure axis ticks based on explicit class labels
        Vector tick_pos(K);
        std::vector<std::string> tick_labels(K);
        for (int ki = 0; ki < K; ++ki) {
            tick_pos[ki]    = static_cast<double>(ki);
            tick_labels[ki] = std::to_string(classes[ki]);
        }
        
        detail::plt::xticks(tick_pos, tick_labels);
        detail::plt::yticks(tick_pos, tick_labels);
        detail::plt::xlabel("Predicted Label");
        detail::plt::ylabel("True Label");
        detail::plt::title(title);
        detail::plt::tight_layout();
        detail::save_and_show(save_path);
    }
};

} // namespace utils
} // namespace sklearn_cpp