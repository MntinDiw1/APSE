#define WITHOUT_NUMPY

#include <iostream>
#include <string>
#include <list>
#include <functional>
#include <cmath>
#include "matplotlibcpp.h"
#include <ranges>
#include <numeric>

namespace plt = matplotlibcpp;

double f1(double x){
    return std::exp(-0.1 * x) * std::sin(2 * x); // f_1 = e^(-0.1x)*sin^(2x)
}

double f2(double x){
    return std::cos(x); // f_2 = cos(x)
}

double f_slow(double x) {
    return std::pow(x - 1, 10); //f_slow = (x - 1)^10
}

double df_calc(std::function<double(double)> f, double x_val){ // i know this isnt how its done but i think it makes more sence to do the differentiation of the system 
    double h = 0.000001; //this si the small delta_x, the smaller the value the more accurate the derivitive
    return (( f(x_val+h) - f(x_val-h) ) / ( 2 * h)); 
}

void newton_raphson(std::function<double(double)> f, double initial_value, int countEq){
    
    int itterations = 100;
    double tolerence = 0.00001;
    double h = 0.000001; //this si the small delta_x, the smaller the value the more accurate the derivitive


    int itt_curr = 0;

    std::list<double> itt_vals = {initial_value};

    bool solved = false;
    
    double x_curr = initial_value;
    double x_prev;
    double df_value; 
    double f_value;

    double diff;    
    double sol;    
    
    while ((itt_curr < itterations) && (solved == false)){        
        x_prev = x_curr;
        
        f_value = f(x_curr);
        df_value = (( f(x_curr+h) - f(x_curr-h) ) / ( 2 * h));
        x_curr = x_prev - (f_value/df_value);
        
        itt_curr++;
        itt_vals.push_back(x_curr);

        diff = std::abs(x_curr - x_prev);
        if ( diff < tolerence){
            solved = true;
            sol = x_curr;
        }
        std::cout << "Itteration " << itt_curr << "; x = " << x_curr << std::endl;
        } 
    if (solved) {
        std::cout << "Solution found at iteration " << itt_curr << "; x = " << sol << std::endl;
        
        std::vector<double> iterations_x;
        for (int i = 0; i <= itt_curr; ++i) iterations_x.push_back(i);


        // code from here f
        std::vector<double> sol_line(iterations_x.size(), sol);

        std::vector<double> itt_vals_vec(itt_vals.begin(), itt_vals.end());

        plt::figure_size(800, 600);
        plt::named_plot("Iteration Values", iterations_x, itt_vals_vec); 
        plt::named_plot("Final Solution", iterations_x, sol_line, "r--"); 
        
        plt::xlabel("Iteration Number");
        plt::ylabel("Value of x");
        plt::title("Newton-Raphson Convergence");
        plt::legend();
        plt::save("./convergence_plot_" + std::to_string(countEq) + ".png");
        plt::show();
    }
    if (!solved){
        std::cout << "No converging solution found after " << itt_curr << std::endl;
    }
}

int main(){
    newton_raphson(f1, 2, 1);
    newton_raphson(f2, 1, 2);
    newton_raphson(f_slow, 2.0, 3);
}