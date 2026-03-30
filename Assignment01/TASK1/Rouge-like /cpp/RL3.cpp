#include <iostream> 

// Structure to define the environment properties
struct room
{
    int hieght{};    // Vertical dimension
    int width {};    // Horizontal dimension
    std::string ground{"*"}; // Floor tile
    std::string corner{"#"}; // Corner boundary
    std::string walls{""};   // Wall property
};

// Structure to define the player state
struct player
{
    int x{};         // Row position
    int y{};         // Column position
    std::string player{"ë"}; // Character avatar
};

// Function to render the 2D grid to the console
void printroom(room R, player playerpos)
{
    // Clear screen simulation (scrolling)
    for (int j = 0; j < 100; ++j){
        std::cout << '\n';
    }

    // Render Top Border
    std::cout << R.corner << " ";
    for (int j = 0; j < R.width; ++j){
         std::cout << "-" << " ";
        }
    std::cout << R.corner << " ";
    std::cout << '\n';

    // Render Grid Rows
    for (int i = 0; i < R.hieght; ++i)
    {
        std::cout << "|" << " "; // Left wall
        for (int j = 0; j < R.width; ++j)
        {
            // Check if current coordinate matches player position
            if (i == playerpos.x && j == playerpos.y)
            {
                std::cout << playerpos.player << " ";
            }
            else
            {
                std::cout << R.ground << " ";
            }
        }
        std::cout << "|" << " "; // Right wall
        std::cout << '\n';
    }

    // Render Bottom Border
    std::cout << R.corner << " ";
    for (int j = 0; j < R.width; ++j){
         std::cout << "-" << " ";
        }
    std::cout << R.corner << " ";
      std::cout << '\n';

}

// Logic for updating player position based on WASD input
void movetheplayer(player &playerpos, char input, room room)
{
    switch (input)
    {
    case 'w':
        if (playerpos.x > 0) // Boundary check: Top
        {
            playerpos.x -= 1;
        }
        break;
    case 's':
        if (playerpos.x < room.hieght - 1) // Boundary check: Bottom
        {
            playerpos.x += 1;
        }
        break;
    case 'a':
        if (playerpos.y > 0) // Boundary check: Left
        {
            playerpos.y -= 1;
        }
        break;
    case 'd':
        if (playerpos.y < room.width - 1) // Boundary check: Right
        {
            playerpos.y += 1;
        }
        break;
    case 'x':
        std::cout << "Exiting the game.\n";
        return exit(0); // Terminate process
    default:
        std::cout << "Invalid input. Use WASD keys to move.\n";
        break;
    }
    std::cout << "Player position: (" << playerpos.x << ", " << playerpos.y << ")\n";
 
}

// User interface for initializing the room dimensions
int selectroomsize(room &room)
{
    int selection {};
    do {
        std::cout << "Select small room  - 1: " << std::endl;
        std::cout << "Select medium room - 2: " << std::endl;
        std::cout << "Select large room  - 3: " << std::endl;
        std::cin >> selection;
    }
    while (selection < 1 || selection > 3);

    switch (selection) {
        case 1:
            room.hieght = 3;
            room.width = 3;
            break;
        case 2:
            room.hieght = 6;
            room.width = 6;
            break;
        case 3:
            room.hieght = 9;
            room.width = 9;
            break;
        default:
            room.hieght = 3;
            room.width = 3;
            break;
    }
}

int main()
{
room room;
selectroomsize(room);
player playerpos{0,0}; // Initialize player at top-left
printroom(room, playerpos);

// Infinite Game Loop
while (true) {
        char input;
        std::cout << "Enter a move (WASD): ";
        std::cin >> input;
movetheplayer(playerpos, input, room);
printroom(room, playerpos);
}
  return 0;
}