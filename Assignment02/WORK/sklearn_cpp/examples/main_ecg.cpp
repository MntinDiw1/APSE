#include <iostream>
#include <vector>

#include "sklearn_cpp/utils/csv_reader.hpp"
#include "sklearn_cpp/utils/visualize.hpp"
#include "sklearn_cpp/linear_model/logistic_regression.hpp"

int main() {
    // Load the ECG dataset
    constexpr auto CSVPATH = "./data/ecg.csv"; // Path to the ECG CSV file (CHANGE THIS TO YOUR PATH)

    std::vector<std::vector<double>> X;
    std::vector<double> y;
    sklearn_cpp::utils::CSVReader::load(CSVPATH, false, X, y);

    std::cout << "Loaded " << X.size() << " samples with " << X[0].size() << " features" << std::endl;

    // Train binary logistic regression - fit() detects 2 classes and runs the binary path
    sklearn_cpp::linear_model::LogisticRegression model;
    model.fit(X, y);

    // Get predictions to feed the plotting helper
    std::vector<int> y_pred = model.predict(X);

    // Report accuracy on the training set
    double accuracy = model.score(X, y);
    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;

    // Plot the confusion matrix - the diagonal cells should hold the largest counts
    sklearn_cpp::utils::Visualizer::plot_confusion_matrix(
        y, y_pred,
        "ECG Anomaly Detection - Confusion Matrix",
        "./plots/ecg_confusion_matrix.png");  // CHANGE THIS TO YOUR DESIRED PATH

    return 0;
}