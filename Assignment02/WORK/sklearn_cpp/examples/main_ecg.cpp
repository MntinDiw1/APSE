#include <iostream>
#include <vector>
// Both of these are only needed for this specific example to do the 80:20 split if only one CSV file is provided, 
// you can remove them if you have separate train and test CSV files
#include <numeric> // for std::iota
#include <random>  // for std::random_device, std::mt19937, std::shuffle

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

    // Check if a separate test CSV was provided, if not split the training data into training and testing sets
    if (std::string(TestCSVPATH) != "") {
        // Load training and testing data from separate CSV files
        //
        sklearn_cpp::utils::CSVReader::load(TrainCSVPATH, false, x_train, y_train);
        std::cout << "Loaded Training data: " << x_train.size() << " samples with " << x_train[0].size() << " features" << std::endl;
        
        sklearn_cpp::utils::CSVReader::load(TestCSVPATH, false, x_test, y_test);
        std::cout << "Loaded Test data: " << x_test.size() << " samples with " << x_test[0].size() << " features" << std::endl;
    } 
    // If there is seperate training and test data the below will not be needed:
    else {
        std::cout << "No separate test CSV provided, using randomized 80:20 split" << std::endl;
        
        // load CSV file
        std::vector<std::vector<double>> x;
        std::vector<double> y;
        sklearn_cpp::utils::CSVReader::load(TrainCSVPATH, false, x, y);
        std::cout << "Loaded data: " << x.size() << " samples with " << x[0].size() << " features" << std::endl;
        
        const std::size_t n_samples = x.size();
        const std::size_t n_train = static_cast<std::size_t>(n_samples * 0.8);
        
        // Create a vector of indices
        std::vector<std::size_t> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
        
        // Shuffle the indices pseudo-randomly
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
        
        // Distribute the data using the shuffled indices
        for (std::size_t i = 0; i < n_train; ++i) {
            x_train.push_back(x[indices[i]]);
            y_train.push_back(y[indices[i]]);
        }
        
        for (std::size_t i = n_train; i < n_samples; ++i) {
            x_test.push_back(x[indices[i]]);
            y_test.push_back(y[indices[i]]);
        }
        
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