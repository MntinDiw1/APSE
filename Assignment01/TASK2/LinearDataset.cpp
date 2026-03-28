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




// Simple Linear Regression using Least Squares
void performLinearRegression(const std::vector<Coordinate>& data, 
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

    std::cout << "\n=== Linear Regression Results ===" << std::endl;
    std::cout << "Fitted slope (k):     " << slope << std::endl;
    std::cout << "Fitted intercept (m): " << intercept << std::endl;
    std::cout << "Original values were: slope = 2.0, intercept = 5.0" << std::endl;

};



int main() {
    double target_k = 2.0;
    double target_m = 5.0;

    DataGenerator mainGen(20, target_k, target_m, 2.0); // Increased noise for visibility
    std::vector<Coordinate> dataset = mainGen.CreateData();

    double fitted_slope = 0.0;
    double fitted_intercept = 0.0;
    performLinearRegression(dataset, fitted_slope, fitted_intercept);

    std::ofstream outFile("LinearDataset.csv");
    if (outFile.is_open()) {
        outFile << "x,y,f_slope,f_intercept\n";
        for (const auto& point : dataset) {
            outFile << point.x << "," << point.y << "," << fitted_slope << "," << fitted_intercept << "\n";
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