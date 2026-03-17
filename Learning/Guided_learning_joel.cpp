#include <vector>
#include <iostream>

// Create a vector of doubles
std::vector<double> x_values; 

// Add an element to the end
x_values.push_back(10.5);

// Create a vector with a specific size (e.g., 100 elements)
std::vector<double> dataset(100);

// Access elements (just like an array)
double first = dataset[0];

// Print the first element
std::cout << "First element: " << first << std::endl;