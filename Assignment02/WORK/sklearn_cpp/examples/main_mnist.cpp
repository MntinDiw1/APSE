#include <iostream>
#include <vector>

#include "sklearn_cpp/utils/csv_reader.hpp"
#include "sklearn_cpp/utils/visualize.hpp"
#include "sklearn_cpp/linear_model/logistic_regression.hpp"

int main() {
    // Load the MNIST dataset
    constexpr auto TrainingCSVpath = "./data/mnist_train.csv"; // Path to the MNIST training CSV file (CHANGE THIS TO YOUR PATH)
    constexpr auto TestCSVpath     = "./data/mnist_test.csv"; // Path to the MNIST test CSV file (CHANGE THIS TO YOUR PATH)

    std::vector<std::vector<double>> x;
    std::vector<double> y;
    std::cout << "Loading Training data..." << std::endl;
    sklearn_cpp::utils::CSVReader::load(TrainingCSVpath, false, x, y);

    std::vector<std::vector<double>> x_test;
    std::vector<double> y_test;
    std::cout << "Loading Test data..." << std::endl;
    sklearn_cpp::utils::CSVReader::load(TestCSVpath, false, x_test, y_test);

    std::cout << "Loaded Training data: " << x.size() << " samples with " << x[0].size() << " features" << std::endl;
    std::cout << "Loaded Test data: " << x_test.size() << " samples with " << x_test[0].size() << " features" << std::endl;

    // Train multi-class logistic regression - fit() detects 10 classes and runs the softmax path
    // Using fewer iterations than the default because the dataset is much bigger
    sklearn_cpp::linear_model::LogisticRegression model(0.1, 500, 0.01);
    std::cout << "Training..." << std::endl;
    model.fit(x, y);

    // Get predictions to feed the plotting helper
    std::vector<int> y_pred = model.predict(x_test);

    // Report accuracy on the test set
    double accuracy = model.score(x_test, y_test);
    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;

    // Plot the 10x10 confusion matrix - most counts should land on the diagonal
    sklearn_cpp::utils::Visualizer::plot_confusion_matrix(
        y_test, y_pred,
        "MNIST Digit Classification - Confusion Matrix",
        "./plots/mnist_confusion_matrix.png");  // CHANGE THIS TO YOUR DESIRED PATH

    // Plot the decision boundary for a 2D cross-section (using central pixels)
    sklearn_cpp::utils::Visualizer::plot_decision_boundary_2d(
        x_test, y_test,
        [&model](const auto& grid_X) { return model.predict(grid_X); },
        "MNIST Decision Boundary (Pixels 350 vs 400)",
        200, 
        "./plots/mnist_decision_boundary.png",
        350, // X-Axis Feature Index (a central pixel)
        400  // Y-Axis Feature Index (another central pixel)
    );
    return 0;
}