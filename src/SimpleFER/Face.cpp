#include "Face.h"
#include <nlohmann/json.hpp>
const std::string Face::getExpressionText(){
    switch (expression)
    {
    case(angry):
        return "angry";
    case(disgust):
        return "disgust";
    case(fear):
        return "fear";
    case(happy):
        return "happy";
    case(sad):
        return "sad";
    case(surprise):
        return "surprise";
    case(neutral):
        return "neutral";
    default:
        return "unknown";
    }
}

std::string Face::serializeFace(const Face& face) {
    using namespace nlohmann;
    json face_json;
    face_json["expression"] = face.expression;
    face_json["x"] = face.region.x;
    face_json["y"] = face.region.y;
    face_json["width"] = face.region.width;
    face_json["height"] = face.region.height;
    return face_json.dump();
}