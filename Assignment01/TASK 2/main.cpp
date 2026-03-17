//write the code to print random numbers between 1 and 100 using C++17
#include <iostream>
#include <random>

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);

    for (int i = 0; i < 10; ++i) {
        std::cout << dis(gen) << " ";
    }
    return 0;
}