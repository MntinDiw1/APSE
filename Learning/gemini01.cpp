#include <iostream>
#include <string>
using namespace std;

// CONST USE CASE: 
// We use 'const' here to ensure MAX_HEALTH is never accidentally changed.
const int MAX_HEALTH = 100; 

// VOID USE CASE:
// This function doesn't need to return a number. It performs an action 
// (modifying the health variable directly), so the return type is 'void'.
void takeDamage(int &hp, int dmg, bool shield) {
    if (shield) {
        cout << "[Defense] Shield active! Damage halved." << endl;
        dmg = dmg / 2; 
    }
    
    // YOUR CODE HERE:
    hp = hp - dmg; 
}


int main() {
    // 1. Setup variables
    string name = "Kael";
    int currentHealth = MAX_HEALTH; // Starts at 100
    bool hasMagicShield = true;
    
    cout << "--- GAME START ---" << endl;
    cout << name << " starts with " << currentHealth << " HP." << endl;
    
    // 2. Simulate an attack

    cout << "Enter damage amount: \n " << endl;
    int damageAmount;
    cin >> damageAmount;
    cout << "A wild goblin attacks for " << damageAmount << " damage!" << endl;
    
    // Passing 'currentHealth' by reference allows the function to lower it.
    takeDamage(currentHealth, damageAmount, hasMagicShield);
    
    // 3. Print the result
    cout << "--- STATUS UPDATE ---" << endl;
    cout << name << " now has " << currentHealth << " HP." << endl;
    
    return 0;
}