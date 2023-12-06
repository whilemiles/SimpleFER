#include <opencv2/core/types.hpp>
enum Emotion
{
    neutral,
    angry,
    happy,
    sad,
    surprised,
    disgusted,
    scared
};

class Face
{
public:
    cv::Rect region;
    Emotion emotion;
    
    const std::string getEmotion()
    {
        //TODO
        if(emotion == 0){
            return "neutral";
        }
        else if (emotion == 1){
            return "angry";
        }
        else if (emotion == 2){
            return "happy";
        }
        else if (emotion == 3){
            return "sad";
        }
        else if (emotion == 4){
            return "surprised";
        }
        else if (emotion == 5){
            return "discusted";
        }
        else{
            return "scared";
        }
    }
};

std::vector<Face> analyzeFace(cv::Mat img);
cv::Mat alignImage(cv::Mat imgGray);
