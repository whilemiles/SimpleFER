#include <iostream>
#include <string>

class User
{
public:
    std::string Name;
    std::string Password;

    User(std::string name, std::string pw)
    {
        Name = name;
        Password = pw;
    }
    static User deserialize(std::string jsonString);
    static int registerUser(User user);
    static int loginUser(User user);
};