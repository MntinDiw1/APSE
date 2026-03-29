// Use Eigen vectors to complete Task 2.

#include<iostream>
#include<vector>
#include<random>
#include<fstream> 
#include "linearregression.h"  // Joel's code for linear regression NO EIGEN VECTORS 
#include "coordinate.h" //idk if this is needed as a header file or if it can just be a struct in the main file, but I put it in a header file for now.
#include "dataset.h" // Will's code for dataset generation 


int main() {
    DataGenerator mainGen(20, 2.0, 5.0, 2.0);  // 20 data points, slope of 2.0, intercept of 5.0, noise range of 2.0
    std::vector<Coordinate> dataset = mainGen.CreateData(); 

    // NORMAL SOLUTION 
    double fitted_slope = 0.0, fitted_intercept = 0.0; 
    // Calling via class scope
    LinearRegression::performLinearRegression(dataset, fitted_slope, fitted_intercept);


    // GRADIENT DESCENT SOLUTION
    double Gfitted_slope = 0.0, Gfitted_intercept = 0.0;
    // Task 2 implementation
    LinearRegression::performGradientDescent(dataset, Gfitted_slope, Gfitted_intercept, 0.001, 5000);

    // Save results to CSV file
    if (outFile.is_open()) {
        outFile << "x,y,f_slope,f_intercept,Gfitted_slope,Gfitted_intercept\n";
        for (const auto& point : dataset) {
            outFile << point.x << "," << point.y << "," 
                    << fitted_slope << "," << fitted_intercept << ","  // kinda convaluted here but it works, just writing the same slope and intercept for each data point, since they are the same for all points
                    << Gfitted_slope << "," << Gfitted_intercept << "\n"; //did this so that all results are in the same file to make it easier to pass to python for plotting
        }
        outFile.close();
        std::cout << "\nLinearDataset.csv created successfully." << std::endl;
    }

    //possibly create new linear regression header file that uses Eigen vectors to perform the same calculations as the current linear regression header file, but using Eigen vectors instead of for loops SAAD CAN DO THIS LOLLOLOLOL. 

    return 0;
}




    /*
    References:
    https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution.html
    
    */