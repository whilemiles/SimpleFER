#include <ATen/core/grad_mode.h>
#include <algorithm>
#include <c10/core/DeviceType.h>
#include <cstdlib>
#include <cmath>
#include <opencv2/core/cvdef.h>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <ostream>
#include <string>
#include "torch/script.h"
#include "torch/torch.h"
#include "FERPipeline.h"
#include "functions.h"
#include <cstdlib>

std::vector<Face> FERPipeline::run(cv::Mat img)
{
    inputImage = img;
    detect();
    align();
    analyze();
    save();
    return faces;
}


void FERPipeline::detect()
{
    faces.clear();
    cv::Mat YuNetOutput;
    detector->setInputSize(inputImage.size());
    detector->detect(inputImage, YuNetOutput);
    if (YuNetOutput.rows > 0) {
        int i = 0;
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
        
        Face face;
        face.region = region;
        face.faceBox = YuNetOutput.row(0);
        faces.push_back(face);
    }
    
}

void FERPipeline::align()
{
    if(faces.empty()){
        return;
    }
    faceRecognizer->alignCrop(inputImage, faces[0].faceBox, alignedImage);
}

// void FERPipeline::align(){
//     if(faces.empty()){
//         return;
//     }
//     float right_eye_x = faces[0].faceBox.at<float>(0, 4);
//     float right_eye_y = faces[0].faceBox.at<float>(0, 5);
//     float left_eye_x = faces[0].faceBox.at<float>(0, 6);
//     float left_eye_y = faces[0].faceBox.at<float>(0, 7);
//     float nose_x = faces[0].faceBox.at<float>(0, 8);
//     float nose_y = faces[0].faceBox.at<float>(0, 9);
//     float mouth_right_x = faces[0].faceBox.at<float>(0, 10);
//     float mouth__right_y = faces[0].faceBox.at<float>(0, 11);
//     float mouth_left_x = faces[0].faceBox.at<float>(0, 12);
//     float mouth__left_y = faces[0].faceBox.at<float>(0, 13);

//     cv::Mat landmarks = (cv::Mat_<float>(5, 2) <<
//         left_eye_x, left_eye_y,
//         right_eye_x, right_eye_y,
//         nose_x, nose_y,
//         mouth_left_x, mouth__left_y,
//         mouth_right_x, mouth__right_y);
    

//     cv::Mat std_landmarks = (cv::Mat_<float>(5, 2) <<
//         38.2946, 51.6963,
//         73.5318, 51.5014,
//         56.0252, 71.7366,
//         41.5493, 92.3655,
//         70.7299, 92.2041);

//     cv::Mat faceImage = inputImage(faces[0].region);

//     //cv::Mat S = std_landmarks;
//     cv::Mat S = std_landmarks;
//     S = S.reshape(0,10);

//     //cv::Mat Q = cv::Mat::ones(5, 3, CV_32F);
//     //landmarks.copyTo(Q(cv::Rect(0, 0, 2, landmarks.rows)));
//     cv::Mat Q = cv::Mat::zeros(10, 4, CV_32F);
//     for(int i = 0; i < 5; i++){
//         int x = landmarks.at<float>(i,0);
//         int y = landmarks.at<float>(i,1);
//         Q.at<float>(i*2, 0) = x;
//         Q.at<float>(i*2, 1) = y;
//         Q.at<float>(i*2, 2) = 1;
//         Q.at<float>(i*2, 3) = 0;
//         Q.at<float>(i*2 + 1, 0) = y;
//         Q.at<float>(i*2 + 1, 1) = -x;
//         Q.at<float>(i*2 + 1, 2) = 0;
//         Q.at<float>(i*2 + 1, 3) = 1;
//     }
    
    
//     // cv::Mat M = (Q.t() * Q).inv() * Q.t() * S;
//     cv::Mat M = (Q.t() * Q).inv() * Q.t() * S;
//     M = M.reshape(0, 2);
    
//     // cv::Mat matrix = M.t();
//     cv::Mat matrix = (cv::Mat_<float>(2, 3) <<
//         M.at<float>(0), M.at<float>(1), M.at<float>(2),
//         -M.at<float>(1), M.at<float>(0), M.at<float>(3));

//     cv::warpAffine(faceImage, alignedImage, matrix, cv::Size(112, 112));
//     //cv::imwrite("aaaa.jpg", faceImage);
//     //cv::imwrite("bbbb.jpg", alignedImage);
// }

// void FERPipeline::align() //6 degree test
// {
//     cv::Mat landmarks = (cv::Mat_<float>(5, 2) <<
//         74, 135,
//         135, 117,
//         107, 160,
//         97, 202,
//         154, 190);

//     cv::Mat std_landmarks = (cv::Mat_<float>(5, 2) <<
//         38.2946, 51.6963,
//         73.5318, 51.5014,
//         56.0252, 71.7366,
//         41.5493, 92.3655,
//         70.7299, 92.2041);

//     cv::Mat face = cv::imread("../misc/face.png");
//     cv::Mat Q = cv::Mat::ones(5, 3, CV_32F);
//     landmarks.copyTo(Q(cv::Rect(0, 0, 2, landmarks.rows)));
//     //std::cout << Q << std::endl;
//     cv::Mat S = std_landmarks;
//     cv::Mat M = (Q.t() * Q).inv() * Q.t() * S;
//     std::cout << M << std::endl;
//     cv::Mat matrix = M.t();
//     std::cout << matrix << std::endl;

//     cv::Mat affine1;
//     cv::warpAffine(face, affine1, matrix, cv::Size(112, 112));

//     cv::Mat affine1_show = affine1.clone();
//     for (int i = 0; i < std_landmarks.rows; ++i) {
//         cv::circle(affine1_show, cv::Point(std_landmarks.at<float>(i, 0), std_landmarks.at<float>(i, 1)), 3, cv::Scalar(0, 255, 0), -1, 16);
//     }

//     cv::imshow("Affine1", affine1);
//     cv::imshow("Affine1_show", affine1_show);
//     cv::waitKey(0);

// }

// void FERPipeline::align() // 4 degree test
// {

//     cv::Mat landmarks = (cv::Mat_<float>(5, 2) <<
//         74, 135,
//         135, 117,
//         107, 160,
//         97, 202,
//         154, 190);


//     cv::Mat std_landmarks = (cv::Mat_<float>(5, 2) <<
//         38.2946, 51.6963,
//         73.5318, 51.5014,
//         56.0252, 71.7366,
//         41.5493, 92.3655,
//         70.7299, 92.2041);

//     cv::Mat face = cv::imread("../misc/face.png");

//     cv::Mat S = std_landmarks;
//     S = S.reshape(0,10);

//     std::cout << S.size << std::endl;
//     std::cout << S << std::endl;

//     cv::Mat Q = cv::Mat::zeros(10, 4, CV_32F);
//     for(int i = 0; i < 5; i++){
//         int x = landmarks.at<float>(i,0);
//         int y = landmarks.at<float>(i,1);
//         //std::cout << x << " "<< y << std::endl;

//         Q.at<float>(i*2, 0) = x;
//         Q.at<float>(i*2, 1) = y;
//         Q.at<float>(i*2, 2) = 1;
//         Q.at<float>(i*2, 3) = 0;
//         Q.at<float>(i*2 + 1, 0) = y;
//         Q.at<float>(i*2 + 1, 1) = -x;
//         Q.at<float>(i*2 + 1, 2) = 0;
//         Q.at<float>(i*2 + 1, 3) = 1;
//     }
//     std::cout << Q << std::endl;
    
    
//     cv::Mat M = (Q.t() * Q).inv() * Q.t() * S;
//     std::cout << M << std::endl;
//     M = M.reshape(0, 2);
//     cv::Mat matrix = (cv::Mat_<float>(2, 3) <<
//         M.at<float>(0), M.at<float>(1), M.at<float>(2),
//         -M.at<float>(1), M.at<float>(0), M.at<float>(3));
//     std::cout << matrix << std::endl;

//     cv::Mat affine1;
//     cv::warpAffine(face, affine1, matrix, cv::Size(112, 112));

//     cv::Mat affine1_show = affine1.clone();
//     for (int i = 0; i < std_landmarks.rows; ++i) {
//         cv::circle(affine1_show, cv::Point(std_landmarks.at<float>(i, 0), std_landmarks.at<float>(i, 1)), 3, cv::Scalar(0, 255, 0), -1, 16);
//     }

//     cv::imshow("Affine1", affine1);
//     cv::imshow("Affine1_show", affine1_show);
//     cv::waitKey(0);
// }

void FERPipeline::analyze()
{
    if(faces.empty()){
        return;
    }
    cv::cvtColor(alignedImage, alignedImage, cv::COLOR_BGR2GRAY);
    Face& face = faces[0];
    //cv::Mat img_face = (alignedImage)(face.region);
    
    cv::Mat img_face_f, img_face_r;

    alignedImage.convertTo(img_face_f, CV_32F, 1.0 / 255);
    cv::resize(img_face_f, img_face_r, {40,40});

    // std::cout << img_face_r.size << std::endl;
    // std::time_t currentTime = std::time(nullptr);
    // std::tm* localTime = std::localtime(&currentTime);
    // char filename[100];
    // std::strftime(filename, sizeof(filename), "%Y-%m-%d_%H-%M-%S.jpg", localTime);
    // cv::imwrite(filename, img_face_r);
    
    at::Tensor img_tensor = torch::from_blob(img_face_r.data, {1, 1, 40, 40}, torch::kFloat32);
    float mean = 0;
    float std = 255;  
    torch::Tensor normalizedTensor = (img_tensor - mean) / std;
    auto input = normalizedTensor.to(torch::kCUDA);
    
    torch::jit::Module model = torch::jit::load("../saved/ResNet18-lr_best.jit");
    model.to(torch::kCUDA);
    
    torch::NoGradGuard no_grad;
    torch::Tensor output = model.forward({input}).toTensor();
    //std::cout << output << std::endl;
    int res = torch::argmax(output, 1).item().toInt();
    //std::cout << res << std::endl;
    face.expression = (Face::Expression)res;
    //std::cout << face.expression << std::endl;
}

void FERPipeline::offline_process(std::string filename)
{
    using namespace cv;
    VideoCapture inputVideo(filename);
    if (!inputVideo.isOpened()) {
        std::cerr << "Error: Unable to open input video file\n";
        return;
    }

    double fps = inputVideo.get(CAP_PROP_FPS);
    Size frameSize(static_cast<int>(inputVideo.get(CAP_PROP_FRAME_WIDTH)),
                   static_cast<int>(inputVideo.get(CAP_PROP_FRAME_HEIGHT)));
    int totalFrames = static_cast<int>(inputVideo.get(CAP_PROP_FRAME_COUNT));
    std::cout<<filename <<std::endl;
    VideoWriter outputVideo(filename + "-fer.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), fps, frameSize);

    if (!outputVideo.isOpened()) {
        std::cerr << "Error: Unable to create output video file\n";
        return;
    }

    Mat frame;
    int frameCount = 0;
    std::vector<Face> faces;

    while (inputVideo.read(frame)) {
        if (frameCount % 2 == 0) {
            faces = this->run(frame);
        }

        for (Face& face : faces)
        {
            cv::rectangle(frame, face.region, cv::Scalar(0, 255, 0), 2);
            std::string expression_text = face.getExpressionText();
            cv::Point text_location{face.region.x, face.region.y - 25};
            cv::putText(frame, expression_text, text_location, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255), 2);
        }

        outputVideo.write(frame);
        frameCount++;
    }

    visualize();
    
    std::cout << "Offline Process Finished" << std::endl;
    inputVideo.release();
    outputVideo.release();
}

void FERPipeline::save()
{
    if(faces.empty()){
        return;
    }
    std::string command = "python ../scripts/sqlite.py " + curUser + " " + faces[0].getExpressionText();
    //std::cout<< command << std::endl;
    system(command.c_str());
}

void FERPipeline::visualize()
{
    std::string command = "python ../scripts/visualize.py " + curUser;
    //std::cout<< command << std::endl;
    system(command.c_str());
}

void FERPipeline::setCurUser(std::string name){
    curUser = name;
}