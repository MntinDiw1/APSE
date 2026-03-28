#include <iostream> 
#include <vector>   
#include <random>

double generateNoise() { 
     // 1. Setup the "Engine" (Mersenne Twister)
 std::random_device rd;  
 std::mt19937 gen(rd()); 

// 2. Define the "Range" (e.g., noise between -0.5 and 0.5)
 std::uniform_real_distribution<double> dist(-0.5, 0.5);

// 3. Generate a number
 double noise = dist(gen);
 return noise;
}




int main() {
    // A matrix with 3 rows and 2 columns
    // Syntax: std::vector<std::vector<type>> name(rows, std::vector<type>(cols));
    int rows = 3;
    int cols = 2;
    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));

    // Filling it with data manually:
    matrix[0][0] = 1.1; // Row 0, Col 0
    matrix[0][1] = 1.2; // Row 0, Col 1
    matrix[1][0] = 2.1; // Row 1, Col 0
    // ... and so on
    
    std::cout << "Value at [1][0]: " << matrix[1][0] << std::endl;

    double noise = generateNoise();

    std::cout << "Generated Noise: " << noise << std::endl;

    return 0;
}

