#include <algorithm>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>

class Face
{
public:
    enum Expression
    {
        angry,
        disgust,
        fear,
        happy,
        neutral,
        sad,
        surprise,
        null
    } expression;
    cv::Rect region;    
    cv::Mat faceBox;
    const std::string getExpressionText();

    static std::string serializeFace(const Face& face);
};