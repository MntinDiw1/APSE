#include <iostream>
#include <vector>
#include <cstdlib>
using namespace std;

void printBoard(const vector<vector<char>>& grid) {
    for (const vector<char>& row : grid) {
        int i = 0;
        for (const char& cell : row) {
            std::cout << cell << " ";
            i++;
            if (i != 3) {
                std::cout << "|"; //only place a divider between cells
            }
        }
        std::cout << "\n";
    }
}

bool winCheck (const vector<vector<char>>& grid, char player) {

    //check rows and columns
    for (int i = 0; i < 3; i++) {
        if(grid[i][0] == player && grid[i][1] == player && grid[i][2] == player || 
           grid[0][i] == player && grid[1][i] == player && grid[2][i] == player) {
            return true;
        }
    } 
    //check diagonals
    if (grid[0][0] == player && grid[1][1] == player && grid[2][2] == player || 
        grid[0][2] == player && grid[1][1] == player && grid[2][0] == player) {
        return true;
    }
    return false;

}

int main() {

    //initialise player
    char player = 'X';

    //initialise a 3x3 grid 
    vector<vector<char>> grid(3, vector<char>(3, ' '));
    while (true) { 

        printBoard(grid);

        int row, col;
        std::cout << "Player " << player << ", enter row and column (R C): ";
        std::cin >> row >> col;

        if (grid[row][col] == ' ') { 

            grid[row][col] = player;

            if (player == 'X') {
                player = 'O';
            } else {
                player = 'X';
            } 

        } else {
            std::cout << "Spot taken. Try again.\n";
            std::cin.ignore(); //clear the input buffer
            std::cin.clear();
        }

        if (winCheck(grid, 'X')) {
            printBoard(grid);
            std::cout << "Crosses win!\n";
            break;
        } else if (winCheck(grid, 'O')) {
            printBoard(grid);
            std::cout << "Noughts win!\n";
            break;
        }
    }

    return 0;
}

// ERROR CHECKING FOR IDIOTS

// grid of dots instead of spaces
// clear previous input from the console
// start again after a win option
// add AI opponent (random)
// ++add reference numbers to the grid for user input
// ++use alphabetic columns and numeric rows for user input

// mention in the report that this code was made from scratch without following the python example, and that the design choices were made based on my own understanding of how to implement a tic-tac-toe game in C++.

// mention in the report that the code does not currently check for a draw condition, which could be added by checking if the grid is full without a winner after each move
// mention in the report that the win check is inefficient and could be optimised by only checking the row, column and diagonals of the most recent move instead of the entire grid each time
// mention in the report that the input validation is very basic and could be improved by checking for out of bounds input and non-integer input, as well as implementing a more robust way to clear the input buffer


