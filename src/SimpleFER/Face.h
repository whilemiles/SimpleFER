#include <opencv2/core/types.hpp>

class Face
{
public:
    enum Expression
    {
        angry,
        disgust,
        fear,
        happy,
        sad,
        surprise,
        neutral
    } expression;
    cv::Rect region;    
    cv::Point left_eye;
    cv::Point right_eye;
    double align_angle;
    const std::string getExpressionText();
};