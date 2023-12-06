#include "analyze.h"
#include "functions.cpp"
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <string>

static int cnt = 0;

std::vector<Face> analyzeFace(cv::Mat img)
{
    std::vector<Face> analyzedFaces;
    std::vector<cv::Rect> faceRegions;
    bool bMultipleFaces = false;
     
    //detect
    cv::CascadeClassifier faceCascade;
    faceCascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml");
    cv::Mat imgGray;
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
    
    faceCascade.detectMultiScale(imgGray, faceRegions,
     1.1, 3, 0, cv::Size(80, 80));

    //align test: 测试结果显示，第一步检测无法显示倾斜的人脸，后续的对齐旋转似乎也不正确。

    // if(faceRegions.size() > 0){
    //     cv::Mat dectectedImage = imgGray(faceRegions[0]);
    //     std::string name = "./" + std::to_string(cnt) + "a.jpg";
    //     std::string name2 = "./" + std::to_string(cnt) + "b.jpg";
    //     if(cnt == 10){
    //         cnt = 0;
    //     }
    //     cv::imwrite(name, dectectedImage);
    //     cv::Mat alianedImage = alignImage(dectectedImage);
    //     cv::imwrite(name2, alianedImage);
    //     cnt++;
    //     cv::namedWindow("T");
    //     cv::imshow("T", alianedImage);
    // }
    Emotion emo[7] ={
        neutral,
        angry,
        happy,
        sad,
        surprised,
        disgusted,
        scared};
    //result
    if(bMultipleFaces){
        for(auto region : faceRegions){
            Face face;
            face.region = region;
            //TODO
            face.emotion = emo[rand() % 7];
            analyzedFaces.push_back(face);
        }
    }
    else{
        Face face;
        face.region = faceRegions[0];
        face.emotion = emo[rand() % 7];
        analyzedFaces.push_back(face);
    }
    
    
    return analyzedFaces;
}

cv::Mat alignImage(cv::Mat faceImage)
{
    cv::Mat alignedImage = faceImage;
    cv::CascadeClassifier eyeCascade;
    eyeCascade.load("/usr/share/opencv4/haarcascades/haarcascade_eye.xml");
    std::vector<cv::Rect> eyeRegions; 
    eyeCascade.detectMultiScale(faceImage, eyeRegions,
     1.1, 3, 0, cv::Size(80, 80));
    for (auto region : eyeRegions){
        cv::rectangle(faceImage, region, cv::Scalar(255, 0, 0), 2);
        cv::Point point{region.x, region.y - 50};
        cv::putText(faceImage, "emotion_text", point, cv::FONT_HERSHEY_PLAIN,
             3, cv::Scalar(255, 0, 0), 2);
    }

    std::sort(eyeRegions.begin(), eyeRegions.end(),
     [](cv::Rect A, cv::Rect B){return A.area() > B.area();});
            cv::Rect leftEye, rightEye;

    if (eyeRegions.size() >= 2){
        cv::Rect eye1 = eyeRegions[0], eye2 = eyeRegions[1];
        if(eye1.x < eye2.x){
            leftEye = eye1;
            rightEye = eye2;
        }
        else{
            leftEye = eye2;
            rightEye = eye1;
        }
    }

    int leftEyeX = leftEye.x + leftEye.width / 2;
    int leftEyeY = leftEye.y + leftEye.height / 2;
    int rightEyeX = rightEye.x + rightEye.width / 2;
    int rightEyeY = rightEye.y + rightEye.height / 2;

    int direction;
    cv::Vec2i Point3rd;
    if(leftEyeY > rightEyeY){
        direction = -1;
        Point3rd = {rightEyeX, leftEyeY};
    }
    else{
        direction = 1;
        Point3rd = {leftEyeX, rightEyeY};
    }
    double a = getEuclideanDistance(leftEye.x, leftEye.y, Point3rd[0], Point3rd[1]);
    double b = getEuclideanDistance(rightEye.x, rightEye.y, Point3rd[0], Point3rd[1]);
    double c = getEuclideanDistance(rightEye.x, rightEye.y, leftEye.x, leftEye.y);
    if(b != 0 && c != 0){
        double cos_a = (b * b + c * c - a * a) / (2 * b * c);
        double angle = std::acos(cos_a);
        angle = angle * 180 / acos(-1);
        if(direction == -1){
            angle = 90 - angle;
        }
        rotateImage(faceImage, alignedImage, angle, 1);
    }
    return alignedImage;
}