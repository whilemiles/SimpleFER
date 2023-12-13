#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <c10/core/DeviceType.h>
#include <cstdlib>
#include <cmath>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <string>
#include "torch/script.h"
#include "torch/torch.h"
#include "analyze.h"
#include "functions.hpp"

static int cnt = 0;

std::vector<Face> analyzeFace(cv::Mat img)
{
    std::vector<Face> analyzedFaces;
    std::vector<cv::Rect> faceRegions;
    bool bDetectMultipleFaces = false;

    //detect
    cv::CascadeClassifier faceCascade;
    faceCascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml");
    cv::Mat imgGray;
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
    
    faceCascade.detectMultiScale(imgGray, faceRegions,
     1.1, 3, 0, cv::Size(80, 80));

    if(faceRegions.size() > 0)
    {
        if (faceRegions.size() > 0) {
            for (auto region : faceRegions) {
                cv::Mat img_face = imgGray(region);
                
                cv::Mat img_face_f, img_face_r;

                img_face.convertTo(img_face_f, CV_32F, 1.0 / 255);
                cv::resize(img_face_f, img_face_r, {48,48});
                at::Tensor img_tensor = torch::from_blob(img_face_r.data, {1, 1, 48, 48}, torch::kFloat32);
                
                auto input = img_tensor.to(torch::kCUDA);
            
                torch::jit::Module model = torch::jit::load("../saved/EmoCNN.jit");
                model.to(torch::kCUDA);
                
                torch::NoGradGuard no_grad;
                auto tmp = model.forward({input});
                torch::Tensor output = model.forward({input}).toTensor();
                std::cout << output << std::endl;
                int res = torch::argmax(output, 1).item().toInt();
                
                Face face;
                face.region = region;
                face.emotion = Emo[res];

                analyzedFaces.push_back(face);

                if(!bDetectMultipleFaces) break;
            }
        }
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