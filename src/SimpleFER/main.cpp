#include "FERPipeline.h"
#include "User.h"
#include "corvusoft/restbed/status_code.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <restbed>
#include <iostream>
#include <fstream>
#include <memory>
#include <iterator>
#include <string>

std::shared_ptr<FERPipeline> pipeline;
int postCount = 0;

void post_handler(const std::shared_ptr<restbed::Session>& session) 
{
    const auto request = session->get_request( );
    size_t content_length = request->get_header("Content-Length", 0);
    auto content_type = request->get_header("Content-Type", "");
    
    if(content_type.compare("image/jpeg") == 0){
        session->fetch
        (   
            content_length, [content_length](const std::shared_ptr<restbed::Session> session, const restbed::Bytes& body)
            {
                cv::Mat img = cv::imdecode(body, cv::IMREAD_COLOR);
                std::vector<Face> faces;
                Face face;
                faces = pipeline->run(img);
                if(!faces.empty())
                {
                    face = faces[0];
                }
                else{
                    face.expression = (Face::Expression)7; //null
                }
                std::string response_body = Face::serializeFace(face);
                std::cout << postCount++ << ": "<< response_body << std::endl;
                restbed::Response response = restbed::Response();
                response.set_status_code(200);
                response.set_body(response_body);
                response.set_header("Content-Type", "application/json");
                session->close(response);
                return;
            }
        );
    }
    else if(content_type.compare("video/mp4") == 0){
        session->fetch
        (   
            content_length, [content_length](const std::shared_ptr<restbed::Session> session, const restbed::Bytes& body)
            {
                std::time_t currentTime = std::time(nullptr);
                std::tm* localTime = std::localtime(&currentTime);
                std::stringstream filename_stream;
                filename_stream << std::put_time(localTime, "%Y-%m-%d_%H-%M-%S") << ".mp4";
                std::string filename = filename_stream.str();
                std::ofstream file(filename, std::ios::out | std::ios::binary);
                if (file.is_open()) {
                    file.write(reinterpret_cast<const char*>(body.data()), body.size());
                    file.close();
                    std::cout << "Video data saved to " << filename << " successfully." << std::endl;
                }
                else {
                    std::cerr << "Error opening file: " << filename << std::endl;
                }
                pipeline->offline_process(filename);
                restbed::Response response = restbed::Response();
                response.set_status_code(200);
                session->close(response);
                return;
            }
        );
    }
}

void register_handler(const std::shared_ptr<restbed::Session> session)
{   
    const auto request = session->get_request();
    size_t content_length = request->get_header("Content-Length", 0);
    auto content_type = request->get_header("Content-Type", "");
    if(content_type.compare("register/json; charset=utf-8") == 0){
        session->fetch(content_length, [](const std::shared_ptr<restbed::Session> session, const restbed::Bytes& body)
        {
            std::string jsonString(body.begin(), body.end());
            User user = User::deserialize(jsonString);
            //std::cout << user.Name << " " << user.Password << std::endl;
            int status = User::registerUser(user);
            std::cout << status << std::endl;

            if(status == 0){
                std::string msg = "Register success";
                session->close(restbed::OK, msg, {{"Content-Length", std::to_string(msg.length())}});
            }
            else if(status == 101){
                std::string msg = "Register failed. Username already exists.";
                session->close(restbed::CONFLICT, msg, {{"Content-Length", std::to_string(msg.length())}});
            }
            else{
                std::string msg = "Register failed. Unknown Error.";
                session->close(restbed::INTERNAL_SERVER_ERROR, msg, {{"Content-Length", std::to_string(msg.length())}});
            }
        });
    }
}

void login_handler(const std::shared_ptr<restbed::Session> session)
{   
    const auto request = session->get_request();
    size_t content_length = request->get_header("Content-Length", 0);
    auto content_type = request->get_header("Content-Type", "");
    if(content_type.compare("login/json; charset=utf-8") == 0){
        session->fetch(content_length, [](const std::shared_ptr<restbed::Session> session, const restbed::Bytes& body)
        {
            std::string jsonString(body.begin(), body.end());
            User user = User::deserialize(jsonString);
            //std::cout << user.Name << " " << user.Password << std::endl;
            int status = User::loginUser(user);
            std::cout << status << std::endl;

            if(status == 0){
                std::string msg = "Login success";
                session->close(restbed::OK, msg, {{"Content-Length", std::to_string(msg.length())}});
            }
            else if(status == 102){
                std::string msg = "Login failed. Invalid username or password.";
                session->close(restbed::UNAUTHORIZED, msg, {{"Content-Length", std::to_string(msg.length())}});
            }
            else{
                std::string msg = "Login failed. Unknown Error.";
                session->close(restbed::INTERNAL_SERVER_ERROR, msg, {{"Content-Length", std::to_string(msg.length())}});
            }
        });
    }
}

void test()
{

}
void execute()
{
    pipeline = std::make_shared<FERPipeline>();

    auto resUpload = std::make_shared<restbed::Resource>();
    resUpload->set_path("/upload");
    resUpload->set_method_handler("POST", post_handler);

    auto resRegister = std::make_shared<restbed::Resource>();
    resRegister->set_path("/register");
    resRegister->set_method_handler("POST", register_handler);

    auto resLogin = std::make_shared<restbed::Resource>();
    resLogin->set_path("/login");
    resLogin->set_method_handler("POST", login_handler);

    auto settings = std::make_shared<restbed::Settings>();
    settings->set_port(5050);

    restbed::Service service;
    service.publish(resUpload);
    service.publish(resRegister);
    service.publish(resLogin);

    service.start(settings);
}

int main(int argc, char* argv[])
{
    execute();
    //test();
    return 0;
}
