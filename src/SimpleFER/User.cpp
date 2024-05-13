#include "User.h"
#include <nlohmann/json.hpp>

int User::registerUser(User user)
{
    std::string command = "python ../scripts/register.py " + user.Name + " " + user.Password;
    //std::cout<< command << std::endl;
    return WEXITSTATUS(system(command.c_str()));
}

int User::loginUser(User user)
{
    std::string command = "python ../scripts/login.py " + user.Name + " " + user.Password;
    //std::cout<< command << std::endl;
    return WEXITSTATUS(system(command.c_str()));
}

User User::deserialize(std::string jsonString)
{
    using namespace nlohmann;

    json userJson = json::parse(jsonString);
    User newUser(userJson["Name"], userJson["Password"]);
    return newUser;
}
