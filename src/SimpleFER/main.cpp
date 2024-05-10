#include "FERPipeline.h"
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
#include <nlohmann/json.hpp>

using namespace nlohmann;
std::shared_ptr<FERPipeline> pipeline;
int postCount = 0;

std::string serializeFace(const Face& face) {
    json face_json;
    face_json["expression"] = face.expression;
    face_json["x"] = face.region.x;
    face_json["y"] = face.region.y;
    face_json["width"] = face.region.width;
    face_json["height"] = face.region.height;
    return face_json.dump();
}

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
                std::string response_body = serializeFace(face);
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

void test()
{
    std::string command = "python ../scripts/visualize.py user";
    system(command.c_str());
}
void execute()
{
    pipeline = std::make_shared<FERPipeline>();

    auto resource = std::make_shared<restbed::Resource>();
    resource->set_path("/upload");
    resource->set_method_handler("POST", post_handler);

    auto settings = std::make_shared<restbed::Settings>();
    settings->set_port(5050);

    restbed::Service service;
    service.publish(resource);
    service.start(settings);
}

int main(int argc, char* argv[])
{
    execute();
    //test();
    return 0;
}
