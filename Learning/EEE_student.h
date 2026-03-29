#include <iostream>
#include <string>      

class EEE_Student {
private:
    std::string name;
    int id;
public:
    EEE_Student(const std::string& name, int id) : name(name), id(id) {}

    void displayInfo() const {
        std::cout << "Name: " << name << ", ID: " << id << std::endl;
    }      
};