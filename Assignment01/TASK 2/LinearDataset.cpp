//NOTE: put into main and seperate into header files etc later on.

#include<iostream>
#include<vector>
#include<random>
#include<fstream> //???

using namespace std; //dont need to keep typing std:: '..'

// Create a struct to bundle coordinate data together
struct Coordinate {
    double x;
    double y;
}

vector<Coordinate> DataGenerator(int n, double k, double m) {
    vector<Coordinate> dataset;
    for (int i=0; i<n; i++) { //i++ or ++i ??????

        // Create data points and add them to dataset
        Coordinate point;
        point.x = i; //+NOISE
        point.y = k * i + m; //+NOISE
        dataset.push_back(point);
    } 

}

//TDL: ADD NOISE TO DATAPOINTS