#include <iostream>
#include <string>
#include <list>
#include <functional>
#include <cmath>

double f1(double x){
    return 0.5 * (x + 5 / x);
}

double f2(double x){
    return std::cos(x);
}

void fixed_point(std::function<double(double)> func, double initial_value){
    
    int itterations = 1000;
    double tolerence = 0.000001;

    int itt_curr = 0;
    std::list<double> itt_vals = {initial_value};
    bool solved = false;
    double x_curr = initial_value;
    double x_prev; 
    double sol;
    double diff;
    
    while ((itt_curr < itterations) && (solved == false)){        
        x_prev = x_curr;
        x_curr = func(x_prev);
        itt_vals.push_back(x_curr);
        itt_curr++;
        diff = std::abs(x_curr - x_prev);
        if ( diff < tolerence){
            solved = true;
            sol = x_curr;
        }
        std::cout << "Itteration " << itt_curr << "; x = " << x_curr << std::endl;
        } 
    if (solved){
        std::cout << "Solution found at itteration " << itt_curr << "; x = " << sol << std::endl;
    }
    if (!solved){
        std::cout << "No converging solution found after " << itt_curr << std::endl;
    }
    return;
}

int main(){
    fixed_point(f1, 2.0);
    fixed_point(f2, 2.0);
}