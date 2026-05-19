#include <iostream> 


struct room
{
    int hieght{};
    int width {};
    std::string ground{"*"};
    std::string corner{"#"};
    std::string walls{""};
};

struct player
{
    int x{};
    int y{};
    std::string player{"ë"};
};

void printroom(room R, player playerpos)
{
    for (int j = 0; j < 100; ++j){
        std::cout << '\n';
    }

    std::cout << R.corner << " ";
    for (int j = 0; j < R.width; ++j){
         std::cout << "-" << " ";
        }
    std::cout << R.corner << " ";
    std::cout << '\n';

    for (int i = 0; i < R.hieght; ++i)
    {
        std::cout << "|" << " ";
        for (int j = 0; j < R.width; ++j)
        {
            if (i == playerpos.x && j == playerpos.y)
            {
                std::cout << playerpos.player << " ";
            }
            else
            {
                std::cout << R.ground << " ";
            }
        }
        std::cout << "|" << " ";
        std::cout << '\n';
    }
    std::cout << R.corner << " ";
    for (int j = 0; j < R.width; ++j){
         std::cout << "-" << " ";
        }
    std::cout << R.corner << " ";
      std::cout << '\n';

}

void movetheplayer(player &playerpos, char input, room room)
{
    switch (input)
    {
    case 'w':
        if (playerpos.x > 0) // Check if moving up is within bounds
        {
            playerpos.x -= 1;
        }
        break;
    case 's':
        if (playerpos.x < room.hieght - 1) // Check if moving down is within bounds
        {
            playerpos.x += 1;
        }
        break;
    case 'a':
        if (playerpos.y > 0) // Check if moving left is within bounds
        {
            playerpos.y -= 1;
        }
        break;
    case 'd':
        if (playerpos.y < room.width - 1) // Check if moving right is within bounds
        {
            playerpos.y += 1;
        }
        break;
    case 'x':
        std::cout << "Exiting the game.\n";
        return exit(0);
    default:
        std::cout << "Invalid input. Use WASD keys to move.\n";
        break;
    }
    std::cout << "Player position: (" << playerpos.x << ", " << playerpos.y << ")\n";
 
}

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
player playerpos{0,0};
printroom(room, playerpos);
while (true) {
        char input;
        std::cout << "Enter a move (WASD): ";
        std::cin >> input;
movetheplayer(playerpos, input, room);
printroom(room, playerpos);
}
  return 0;
}
