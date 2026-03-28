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
void performLinearRegression(const std::vector<Coordinate>& data) {
    if (data.empty()) {
        std::cout << "No data for regression." << std::endl;
        return;
    }

    double sumX = 0.0, sumY = 0.0, sumXY = 0.0, sumX2 = 0.0;
    int n = data.size();

    for (const auto& point : data) {
        sumX += point.x;
        sumY += point.y;
        sumXY += point.x * point.y;
        sumX2 += point.x * point.x;
    }

    double denominator = n * sumX2 - sumX * sumX;
    if (std::abs(denominator) < 1e-10) {
        std::cout << "Warning: Cannot compute slope (degenerate data)." << std::endl;
        return;
    }

    double slope = (n * sumXY - sumX * sumY) / denominator;
    double intercept = (sumY - slope * sumX) / n;

    std::cout << "\n=== Linear Regression Results ===" << std::endl;
    std::cout << "Fitted slope (k):     " << slope << std::endl;
    std::cout << "Fitted intercept (m): " << intercept << std::endl;
    std::cout << "Original values were: slope = 2.0, intercept = 5.0" << std::endl;
}




int main() {

    DataGenerator mainGen(20, 2, 5, 1);
    std::vector<Coordinate> dataset = mainGen.CreateData(); 

    // Write dataset to CSV file
    std::ofstream outFile("LinearDataset.csv");
    for (const auto& point : dataset) {
        outFile << point.x << "," << point.y << "\n";
    }
    outFile.close();
    std::cout << "\n.csv FILE CREATED" << std::endl;

    performLinearRegression(dataset);
  
}



    /*
    References:
    https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution.html
    
    */