//NOTE: put into main and seperate into header files etc later on.
#include<iostream>
#include<vector>
#include<random>
#include<fstream> 


// Create a struct to bundle coordinate data together
struct Coordinate {
    double x;
    double y;
};

class DataGenerator {
    private:
        int n; //number of data points
        double k; //actual slope
        double m; //actual y-intercept 
        std::mt19937 gen; //random number generator must be a private member variable so that it doesn't die when the function ends
        std::uniform_real_distribution<double> distrib; // distribution range for noise must be a private member

    public:
        DataGenerator(int points, double slope , double intercept, double noise_range) { //constructor
            n = points;
            k = slope;
            m = intercept;
            std::random_device rd; //seed for random number generator
            gen.seed(rd());
            distrib = std::uniform_real_distribution<double>(-0.5*noise_range, 0.5*noise_range);            

        }
        std::vector<Coordinate> CreateData() {
            std::vector<Coordinate> dataset;
                for(int i=0; i<n; ++i) {

                    // Create data points and add them to dataset
                    Coordinate point;
                    point.x = i + distrib(gen); //x + noise
                    point.y = k * i + m + distrib(gen); // y + noise
                    dataset.push_back(point);
                }    

            //print the dataset to console, for testing purposes
            std::cout << "DATASET GENERATED:" << std::endl;
            for(const auto& point : dataset) {
                std::cout << "x: " << point.x << ", y: " << point.y << std::endl;
            }

             return dataset;
        }
};

class LinearRegression {
public:
    // Task 1: Analytical Solution (OLS)
    static void performLinearRegression(const std::vector<Coordinate>& data, 
                                         double& slope, double& intercept) {
        double sumX = 0.0, sumY = 0.0, sumXY = 0.0, sumX2 = 0.0; 
        int n = data.size();

        for (const auto& point : data) {
            sumX += point.x;
            sumY += point.y;
            sumXY += point.x * point.y;
            sumX2 += point.x * point.x;
        }

        slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        intercept = (sumY - slope * sumX) / n;

        std::cout << "\n=== OLS Results ===\nSlope: " << slope << "\nIntercept: " << intercept << std::endl;
    }

    // Task 2: Iterative Solution (Gradient Descent)
    static void performGradientDescent(const std::vector<Coordinate>& data, 
                                        double& w, double& b, 
                                        double learning_rate, int epochs) {
        int n = data.size();
        w = 0.0; b = 0.0; 

        for (int i = 0; i < epochs; ++i) {
            double dL_dw = 0.0;
            double dL_db = 0.0;

            for (const auto& point : data) {
                double prediction = (w * point.x) + b;
                double error = prediction - point.y;
                
                // Derivatives from Equation 7 and 8
                dL_dw += error * point.x;
                dL_db += error;
            }

            // Update parameters: Subtract the gradient
            w -= (2.0 / n) * learning_rate * dL_dw;
            b -= (2.0 / n) * learning_rate * dL_db;
        }
        
        std::cout << "\n=== GD Results ===\nSlope: " << w << "\nIntercept: " << b << std::endl;
    }
};

int main() {
    DataGenerator mainGen(20, 2.0, 5.0, 2.0);
    std::vector<Coordinate> dataset = mainGen.CreateData();

    double fitted_slope = 0.0, fitted_intercept = 0.0;
    // Calling via class scope
    LinearRegression::performLinearRegression(dataset, fitted_slope, fitted_intercept);

    double Gfitted_slope = 0.0, Gfitted_intercept = 0.0;
    // Task 2 implementation
    LinearRegression::performGradientDescent(dataset, Gfitted_slope, Gfitted_intercept, 0.001, 5000);

    std::ofstream outFile("LinearDataset.csv");
    if (outFile.is_open()) {
        outFile << "x,y,f_slope,f_intercept,Gfitted_slope,Gfitted_intercept\n";
        for (const auto& point : dataset) {
            outFile << point.x << "," << point.y << "," 
                    << fitted_slope << "," << fitted_intercept << "," 
                    << Gfitted_slope << "," << Gfitted_intercept << "\n";
        }
        outFile.close();
        std::cout << "\nLinearDataset.csv created successfully." << std::endl;
    }

    return 0;
}




    /*
    References:
    https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution.html
    
    */