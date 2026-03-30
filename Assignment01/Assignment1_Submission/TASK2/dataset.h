#ifndef DATASET_H
#define DATASET_H
#include <vector>
#include <random>
#include <fstream>
#include "coordinate.h"
#include "linearregression.h"

class DataGenerator { // Class to generate datasets for linear regression
    private:
        int n; // number of data points
        double m1; // slope for x1
        double m2; // slope for x2
        double b; // intercept
        std::mt19937 gen; // Mersenne Twister random number generator
        std::uniform_real_distribution<double> distrib; // Uniform distribution for noise

    public:
        // Generate a multipoint dataset with two x values and one y value
        void multipointGen(int points, double slope1, double slope2, double intercept, double noise_range) { // Method to generate a multipoint dataset with two x values and one y value
            n = points; // Set the number of data points
            m1 = slope1; // Set slope for x1
            m2 = slope2; // Set slope for x2
            b = intercept; // Set intercept
            std::random_device rd; // Random device to seed the generator
            gen.seed(rd()); // Seed the generator
            distrib = std::uniform_real_distribution<double>(-0.5 * noise_range, 0.5 * noise_range); // Noise in the range [-noise_range/2, noise_range/2]
        }

        std::vector<coordinate> CreateMultipointData() { // Method to create the multipoint dataset based on the parameters set in multipointGen
            std::vector<coordinate> dataset;
            for(int i = 0; i < n; ++i) { // Generate n data points
                coordinate point;
                
                double x1_val = i + distrib(gen);
                double x2_val = (i * 0.5) + distrib(gen);
                
                point.x.push_back(x1_val);
                point.x.push_back(x2_val);
                
                point.y = (m1 * x1_val) + (m2 * x2_val) + b + distrib(gen); 
                std::cout << "Generated point: (" << x1_val << ", " << x2_val << ") -> " << point.y << std::endl; // Debug print to verify generated points
                dataset.push_back(point);
            }    
            return dataset; // Return the generated dataset
        }
        // Genrate single point dataset with one x value and one y value
        void singlepointGen(int points, double slope1, double intercept, double noise_range) { 
            n = points; // Set the number of data points
            m1 = slope1; // Set slope for x1
            b = intercept; // Set intercept
            std::random_device rd; // Random device to seed the generator
            gen.seed(rd()); // Seed the generator
            distrib = std::uniform_real_distribution<double>(-0.5 * noise_range, 0.5 * noise_range); // Noise in the range [-noise_range/2, noise_range/2]
        }

        std::vector<coordinate> CreateSinglepointData() { // Method to create the single point dataset based on the parameters set in singlepointGen
            std::vector<coordinate> dataset;
            for(int i = 0; i < n; ++i) { // Generate n data points
                coordinate point;
                
                double x1_val = i + distrib(gen);
                point.x.push_back(x1_val);
                
                point.y = (m1 * x1_val) + b + distrib(gen); 
                std::cout << "Generated point: (" << x1_val << ") -> " << point.y << std::endl; // Debug print to verify generated points
                dataset.push_back(point);
            }    
            return dataset; // Return the generated dataset
        }
};
#endif