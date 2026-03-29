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




// Simple Linear Regression using Least Squares, NO EIGAN VALUES
void performLinearRegression(const std::vector<Coordinate>& data, 
                             double& slope, double& intercept) {

    double sumX = 0.0, sumY = 0.0, sumXY = 0.0, sumX2 = 0.0; 
    int n = data.size(); // number of data points

    for (const auto& point : data) {
        sumX += point.x; // sigma x i=1 to m
        sumY += point.y; // sigma y i=1 to m
        sumXY += point.x * point.y; //  sigma x*y i=1 to m
        sumX2 += point.x * point.x; // sigma x^2 i=1 to m
    }


    slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX); // w*
    intercept = (sumY - slope * sumX) / n; // b*

    std::cout << "\n=== Linear Regression Results ===" << std::endl;
    std::cout << "Fitted slope (k):     " << slope << std::endl;
    std::cout << "Fitted intercept (m): " << intercept << std::endl;
    std::cout << "Original values were: slope = 2.0, intercept = 5.0" << std::endl;

};

void performGradientDescent(const std::vector<Coordinate>& data, 
                            double& w, double& b, 
                            double learning_rate, int epochs) {
    int n = data.size();
    
    // Initial values (Start at the top of the mountain)
    w = 0.0;
    b = 0.0;

    for (int i = 0; i < epochs; ++i) {
        double dL_dw = 0.0; // The gradient for slope
        double dL_db = 0.0; // The gradient for intercept

        // 1. Calculate the TOTAL gradient for this step
        for (const auto& point : data) {
            double prediction = (w * point.x) + b;
            double error = prediction - point.y;

            // These formulas come directly from your Equation 7 and 8
            dL_dw += error * point.x;
            dL_db += error;
        }

        // 2. Update the parameters (The "Step" downhill)
        // We multiply by 2/n to get the average derivative of the MSE
        w -= (1.0 / n) * learning_rate * dL_dw;
        b -= (1.0 / n) * learning_rate * dL_db;
    }
     
    std::cout << "\n=== Linear Regression Results ===" << std::endl;
    std::cout << "Fitted slope (k):     " << w << std::endl;
    std::cout << "Fitted intercept (m): " << b << std::endl;
    std::cout << "Original values were: slope = 2.0, intercept = 5.0" << std::endl;
}







int main() {
    double target_k = 2.0;
    double target_m = 5.0;

    DataGenerator mainGen(20, target_k, target_m, 2.0); // Increased noise for visibility
    std::vector<Coordinate> dataset = mainGen.CreateData();

    double fitted_slope = 0.0;
    double fitted_intercept = 0.0;
    performLinearRegression(dataset, fitted_slope, fitted_intercept);


    double Gfitted_slope = 0.0;
    double Gfitted_intercept = 0.0;
    performGradientDescent(dataset, Gfitted_slope, Gfitted_intercept, .01, 100);

    std::ofstream outFile("LinearDataset.csv");
    if (outFile.is_open()) {
        outFile << "x,y,f_slope,f_intercept,Gfitted_slope,Gfitted_intercept\n";
        for (const auto& point : dataset) {
            outFile << point.x << "," << point.y << "," << fitted_slope << "," << fitted_intercept << "," << Gfitted_slope << "," << Gfitted_intercept << "\n";
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