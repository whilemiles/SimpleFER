#include "Face.h"

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