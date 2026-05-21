#include <iostream>
#include <vector>

#include "sklearn_cpp/utils/csv_reader.hpp"
#include "sklearn_cpp/utils/visualize.hpp"
#include "sklearn_cpp/linear_model/logistic_regression.hpp"

int main() {
    // Load the MNIST dataset
    constexpr auto CSVPATH = "./data/mnist_micro.csv"; // Path to the MNIST CSV file (CHANGE THIS TO YOUR PATH)

    std::vector<std::vector<double>> x;
    std::vector<double> y;
    std::cout << "Loading data...\n";
    sklearn_cpp::utils::CSVReader::load(CSVPATH, false, x, y);

    std::cout << "Loaded " << x.size() << " samples with " << x[0].size() << " features" << std::endl;

    // Train multi-class logistic regression - fit() detects 10 classes and runs the softmax path
    // Using fewer iterations than the default because the dataset is much bigger
    sklearn_cpp::linear_model::LogisticRegression model(0.1, 500, 0.01);
    std::cout << "Training...\n";
    model.fit(x, y);

    // Get predictions to feed the plotting helper
    std::vector<int> y_pred = model.predict(x);

    // Report accuracy on the training set
    double accuracy = model.score(x, y);
    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;

    // Plot the 10x10 confusion matrix - most counts should land on the diagonal
    sklearn_cpp::utils::Visualizer::plot_confusion_matrix(
        y, y_pred,
        "MNIST Digit Classification - Confusion Matrix",
        "./plots/mnist_confusion_matrix.png");  // CHANGE THIS TO YOUR DESIRED PATH

    return 0;
}