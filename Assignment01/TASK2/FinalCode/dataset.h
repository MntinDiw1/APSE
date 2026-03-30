#ifndef DATASET_H
#define DATASET_H
#include<vector>
#include<random>
#include<fstream>
#include "coordinate.h"
#include "linearregression.h"



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

#endif // DATASET_H