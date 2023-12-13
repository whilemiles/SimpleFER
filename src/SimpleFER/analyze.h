#include <iostream>
#include <opencv2/core/types.hpp>

enum Emotion
{
    angry,
    disgust,
    fear,
    happy,
    sad,
    surprise,
    neutral
} Emo[7] = {
    angry,
    disgust,
    fear,
    happy,
    sad,
    surprise,
    neutral
};

std::string Enum2String(Emotion e)
{
    switch (e)
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


class Face
{
public:
    cv::Rect region;
    Emotion emotion;
    
    const std::string getEmotion()
    {
        return Enum2String(emotion);
    }
};

std::vector<Face> analyzeFace(cv::Mat img);
cv::Mat alignImage(cv::Mat imgGray);
