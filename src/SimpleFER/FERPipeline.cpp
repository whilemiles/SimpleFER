#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <c10/core/DeviceType.h>
#include <cstdlib>
#include <cmath>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <string>
#include "torch/script.h"
#include "torch/torch.h"
#include "FERPipeline.h"
#include "functions.h"

static int cnt = 0;

std::vector<Face> FERPipeline(cv::Mat img)
{

    //detect
    auto detectedFaces = detectFace(img);

    //align
    auto alignedFaces = alignFace(img, detectedFaces);

    // TODO::normalize
    
    
    //analyze
    auto analyzedFaces = analyzeFace(img, alignedFaces);
    
    return analyzedFaces;
}


std::vector<Face> detectFace(cv::Mat img)
{
    bool bDetectMultipleFaces = false;

    std::vector<Face> detectedFaces;

    cv::Mat YuNetOutput;
    cv::Ptr<cv::FaceDetectorYN> detector = cv::FaceDetectorYN::create("../saved/yunet.onnx", "", img.size());
    detector->detect(img, YuNetOutput);

    if (YuNetOutput.rows > 0) {
        for (int i = 0; i < YuNetOutput.rows; i++) {
            float x, y, w, h;
            x = std::max<float>(0, YuNetOutput.at<float>(i, 0));
            y = std::max<float>(0, YuNetOutput.at<float>(i, 1));
            w = std::max<float>(0, YuNetOutput.at<float>(i, 2));
            h = std::max<float>(0, YuNetOutput.at<float>(i, 3));

            if (x + w >= img.cols)
            w = img.cols - x;
            if (y + h >= img.rows)
            h = img.rows - y;

            cv::Rect region(x, y, w, h);
            
            cv::Point right_eye(std::max<float>(0, YuNetOutput.at<float>(i, 4)), std::max<float>(0, YuNetOutput.at<float>(i, 5)));
            cv::Point left_eye(std::max<float>(0, YuNetOutput.at<float>(i, 6)), std::max<float>(0, YuNetOutput.at<float>(i, 7)));
            
            Face face;
            face.region = region;
            face.right_eye = right_eye;
            face.left_eye = left_eye;
            detectedFaces.push_back(face);
            if (!bDetectMultipleFaces) break;
        }
    }
    return detectedFaces;
}

std::vector<Face> alignFace(cv::Mat img, std::vector<Face> faces)
{
    std::vector<Face> alingnedFaces;
    for(auto face : faces){
        int direction;
        cv::Point Point3rd;
        if(face.left_eye.y > face.right_eye.y){
            Point3rd = {face.right_eye.x, face.left_eye.y};
            direction = -1;
        }
        else{
            Point3rd = {face.left_eye.x, face.right_eye.y};
            direction = 1;
        }
        double a = getEuclideanDistance(face.left_eye, Point3rd);
        double b = getEuclideanDistance(face.right_eye, Point3rd);
        double c = getEuclideanDistance(face.right_eye, face.left_eye);
        if(b != 0 && c != 0){
            double cos_a = (b * b + c * c - a * a) / (2 * b * c);
            double angle = std::acos(cos_a);
            angle = angle * 180 / acos(-1);
            if(direction == -1){
                angle = 90 - angle;
            }
            else{
                angle = -angle;
            }
            face.align_angle = angle;
            alingnedFaces.push_back(face);
        }
    }
    return alingnedFaces;
}

std::vector<Face> analyzeFace(cv::Mat img, std::vector<Face> faces)
{
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    std::vector<Face> analyzedFaces;
    for(auto face : faces){
        cv::Mat img_face = img(face.region);
        
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
        int res = torch::argmax(output, 1).item().toInt();
        
        face.emotion = (Face::Emotion)res;

        analyzedFaces.push_back(face);
    }
    return analyzedFaces;
}

