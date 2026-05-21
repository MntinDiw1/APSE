#include <iostream>
#include <vector>

#include "sklearn_cpp/utils/csv_reader.hpp"
#include "sklearn_cpp/utils/visualize.hpp"
#include "sklearn_cpp/linear_model/logistic_regression.hpp"

int main() {
    // Load the ECG dataset
    constexpr auto TrainCSVPATH = "./data/ecg.csv"; // Path to the ECG CSV file (CHANGE THIS TO YOUR PATH)
    // Leave TestCSVPATH as empty string if you only have one CSV file with all the data, the programme will split 80:20 for training and testing
    constexpr auto TestCSVPATH = ""; // Path to the ECG test CSV file (CHANGE THIS TO YOUR PATH)
    
    std::vector<std::vector<double>> x_train;
    std::vector<double> y_train;
    std::vector<std::vector<double>> x_test;
    std::vector<double> y_test;

    std::vector<std::vector<double>> x_train;
    std::vector<double> y_train;
    std::vector<std::vector<double>> x_test;
    std::vector<double> y_test;

    // Check if a separate test CSV was provided, if not split the training data into training and testing sets
    if (TestCSVPATH != "") {
        // REMOVED the "std::vector<...>" type declarations to prevent shadowing
        sklearn_cpp::utils::CSVReader::load(TrainCSVPATH, false, x_train, y_train);
        std::cout << "Loaded Training data: " << x_train.size() << " samples with " << x_train[0].size() << " features" << std::endl;
        
        sklearn_cpp::utils::CSVReader::load(TestCSVPATH, false, x_test, y_test);
        std::cout << "Loaded Test data: " << x_test.size() << " samples with " << x_test[0].size() << " features" << std::endl;
    } else {
        std::cout << "No separate test CSV provided, using 80:20 split for training and testing" << std::endl;
        // load CSV file
        std::vector<std::vector<double>> x;
        std::vector<double> y;
        sklearn_cpp::utils::CSVReader::load(TrainCSVPATH, false, x, y);
        std::cout << "Loaded data: " << x.size() << " samples with " << x[0].size() << " features" << std::endl;
        
        // Split into training and testing sets
        const std::size_t n_samples = x.size();
        const std::size_t n_train = static_cast<std::size_t>(n_samples * 0.8);
        
        // USE .assign() to populate the outer variables!
        x_train.assign(x.begin(), x.begin() + n_train);
        y_train.assign(y.begin(), y.begin() + n_train);
        
        x_test.assign(x.begin() + n_train, x.end());
        y_test.assign(y.begin() + n_train, y.end());
        
        std::cout << "Training set: " << x_train.size() << " samples with " << x_train[0].size() << " features" << std::endl;
        std::cout << "Test set: " << x_test.size() << " samples with " << x_test[0].size() << " features" << std::endl;
    }
    
    
    // Train binary logistic regression - fit() detects 2 classes and runs the binary path
    sklearn_cpp::linear_model::LogisticRegression model;
    std::cout << "Training..." << std::endl;
    model.fit(x_train, y_train);

    // Get predictions to feed the plotting helper
    std::cout << "Testing..." << std::endl;
    std::vector<int> y_pred = model.predict(x_test);

    // Report accuracy on the training set
    double accuracy = model.score(x_test, y_test);
    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;

    // Plot the confusion matrix - the diagonal cells should hold the largest counts
    sklearn_cpp::utils::Visualizer::plot_confusion_matrix(
        y_test, y_pred,
        "ECG Anomaly Detection - Confusion Matrix",
        "./plots/ecg_confusion_matrix.png");  // CHANGE THIS TO YOUR DESIRED PATH

    return 0;
}