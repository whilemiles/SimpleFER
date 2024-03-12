#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <c10/core/DeviceType.h>
#include <cstdlib>
#include <cmath>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/persistence.hpp>
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

std::vector<Face> FERPipeline::run(cv::Mat img)
{
    inputImage = img;
    //detect
    detect();
    //align
    //align();
    // TODO::normalize
    
    //analyze
    analyze();
    
    return faces;
}


void FERPipeline::detect()
{
    faces.clear();
    bool bDetectMultipleFaces = false;
    cv::Mat YuNetOutput;
    cv::Ptr<cv::FaceDetectorYN> detector = cv::FaceDetectorYN::create("../saved/yunet.onnx", "", inputImage.size());
    detector->detect(inputImage, YuNetOutput);

    if (YuNetOutput.rows > 0) {
        for (int i = 0; i < YuNetOutput.rows; i++) {
            float x, y, w, h;
            x = std::max<float>(0, YuNetOutput.at<float>(i, 0));
            y = std::max<float>(0, YuNetOutput.at<float>(i, 1));
            w = std::max<float>(0, YuNetOutput.at<float>(i, 2));
            h = std::max<float>(0, YuNetOutput.at<float>(i, 3));

            if (x + w >= inputImage.cols) 
            w = inputImage.cols - x;
            if (y + h >= inputImage.rows)
            h = inputImage.rows - y;

            cv::Rect region(x, y, w, h);
            
            cv::Point right_eye(std::max<float>(0, YuNetOutput.at<float>(i, 4)), std::max<float>(0, YuNetOutput.at<float>(i, 5)));
            cv::Point left_eye(std::max<float>(0, YuNetOutput.at<float>(i, 6)), std::max<float>(0, YuNetOutput.at<float>(i, 7)));
            
            Face face;
            face.region = region;
            face.right_eye = right_eye;
            face.left_eye = left_eye;
            faces.push_back(face);
            if (!bDetectMultipleFaces) break;
        }
    }
    
}

void FERPipeline::align()
{
    for(Face& face : faces){
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
        }
    }
}

void FERPipeline::analyze()
{
    if(faces.empty()){
        return;
    }
    cv::cvtColor(inputImage, inputImage, cv::COLOR_BGR2GRAY);
    for(Face& face : faces){
        cv::Mat img_face = (inputImage)(face.region);
        
        cv::Mat img_face_f, img_face_r;

        img_face.convertTo(img_face_f, CV_32F, 1.0 / 255);
        cv::resize(img_face_f, img_face_r, {48,48});
        at::Tensor img_tensor = torch::from_blob(img_face_r.data, {1, 1, 48, 48}, torch::kFloat32);
        float mean = 0; 
        float std = 255;  
        torch::Tensor normalizedTensor = (img_tensor - mean) / std;
        auto input = normalizedTensor.to(torch::kCUDA);
        
        torch::jit::Module model = torch::jit::load("../saved/ResNet18_best.jit");
        //torch::jit::Module model = torch::jit::load("../saved/acc73.jit");
        model.to(torch::kCUDA);
        
        torch::NoGradGuard no_grad;
        torch::Tensor output = model.forward({input}).toTensor();
        int res = torch::argmax(output, 1).item().toInt();
        face.expression = (Face::Expression)res;
    }
}

