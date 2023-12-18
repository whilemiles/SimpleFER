#include <opencv2/core/types.hpp>

class Face
{
public:
    enum Emotion
    {
        angry,
        disgust,
        fear,
        happy,
        sad,
        surprise,
        neutral
    } emotion;
    cv::Rect region;    
    
    const std::string getEmotionText();
};