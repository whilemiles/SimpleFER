#include "FERPipeline.h"
#include "User.h"
#include "corvusoft/restbed/response.hpp"
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
#include <filesystem>

std::shared_ptr<FERPipeline> pipeline;
int postCount = 0;
namespace fs = std::filesystem;

void post_handler(const std::shared_ptr<restbed::Session>& session) 
{
    const auto request = session->get_request( );
    size_t content_length = request->get_header("Content-Length", 0);
    auto content_type = request->get_header("Content-Type", "");
    auto query_parameters = request->get_query_parameters();
    std::string user_name = "";
    for (const auto& parameter : query_parameters) {
        if (parameter.first == "user_name") {
            user_name = parameter.second;
            break;
        }
    }
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
            content_length, [content_length, user_name](const std::shared_ptr<restbed::Session> session, const restbed::Bytes& body)
            {
                std::time_t currentTime = std::time(nullptr);
                std::tm* localTime = std::localtime(&currentTime);
                std::stringstream filename_stream;
                filename_stream << std::put_time(localTime, "%Y-%m-%d_%H-%M-%S") << "-" << user_name << ".mp4";
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
                pipeline->setCurUser(user_name);
                pipeline->offline_process(filename);
                
                std::ifstream new_video_file(filename + "-fer.mp4", std::ios::binary | std::ios::ate);
                std::streamsize size = new_video_file.tellg();
                new_video_file.seekg(0, std::ios::beg);

                std::vector<char> buffer(size);
                if (new_video_file.read(buffer.data(), size)) {
                    session->close(restbed::OK, std::string(buffer.data(), buffer.size()), { { "Content-Length", std::to_string(size) }, { "Content-Type", "video/mp4" } });
                } else {
                    session->close(restbed::INTERNAL_SERVER_ERROR);
                }
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
            //std::cout << status << std::endl;

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
    std::cout << "login handler" << std::endl;
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
            //std::cout << status << std::endl;

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

void get_image_urls(const std::shared_ptr<restbed::Session>& session) {
    const auto request = session->get_request( );

    auto query_parameters = request->get_query_parameters();
    std::string user_name = "";
    for (const auto& parameter : query_parameters) {
        if (parameter.first == "user_name") {
            user_name = parameter.second;
            break;
        }
    }

    std::stringstream response_body;

    std::string url_base = "http://172.21.117.218:5050/image?path=visual_pics/" + user_name + "/";
    std::string folderPath = "./visual_pics/" + user_name;
    if (fs::exists(folderPath) && fs::is_directory(folderPath)) {
        for (const auto& entry : fs::directory_iterator(folderPath)) {
            if (entry.is_regular_file()) {
                response_body << url_base << entry.path().filename().string().c_str() << " "; 
            }
        }
    } else {
        std::cout << "文件夹不存在或无权访问" << std::endl;
    }
    //std::cout << response_body.str() << std::endl;
    const auto response = std::make_shared<restbed::Response>( );

    response->set_status_code(200);
    response->set_header("Content-Type", "text/plain");
    response->set_body(response_body.str());
    session->close(*response);
}

void get_image(const std::shared_ptr<restbed::Session>& session) {
    std::cout << "get image" << std::endl;

    const auto request = session->get_request( );

    auto query_parameters = request->get_query_parameters();
    std::string path = "";
    for (const auto& parameter : query_parameters) {
        if (parameter.first == "path") {
            path = parameter.second;
            break;
        }
    }

    std::cout << "path:"<< path << std::endl;
    // session->fetch(path, [](const std::shared_ptr<restbed::Session> session, const restbed::Bytes &body)
    // {
    //     session->close(restbed::OK, body);
    // });



    const auto response = std::make_shared<restbed::Response>( );
    response->set_header("Content-Type", "image/jpeg");
    std::ifstream image_file(path, std::ios::binary | std::ios::ate);
    if (!image_file.is_open()) {
        response->set_status_code(404);
        session->close(*response);
        return;
    }

    const auto image_size = image_file.tellg();
    image_file.seekg(0, std::ios::beg);

    std::vector<uint8_t> image_data(image_size);
    image_file.read(reinterpret_cast<char*>(image_data.data()), image_size);

    response->set_status_code(200);
    response->set_body(image_data);
    session->close(*response);
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

    auto resImage = std::make_shared<restbed::Resource>();
    resImage->set_path("/image");
    resImage->set_method_handler("GET", get_image);
    

    auto resImageUrls = std::make_shared<restbed::Resource>();
    resImageUrls->set_path("/image-urls");
    resImageUrls->set_method_handler("GET", get_image_urls);

    auto settings = std::make_shared<restbed::Settings>();
    //settings->set_root("./");
    settings->set_port(5050);

    restbed::Service service;
    service.publish(resUpload);
    service.publish(resRegister);
    service.publish(resLogin);
    service.publish(resImage);
    service.publish(resImageUrls);

    service.start(settings);
}

int main(int argc, char* argv[])
{
    execute();
    //test();
    return 0;
}
