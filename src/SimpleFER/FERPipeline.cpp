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
    
    // std::time_t currentTime = std::time(nullptr);
    // std::tm* localTime = std::localtime(&currentTime);
    // char filename[100];
    // std::strftime(filename, sizeof(filename), "%Y-%m-%d_%H-%M-%S.jpg", localTime);
    // cv::imwrite(filename, alignedImage);
}

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
    FERPipeline pipeline;
    std::vector<Face> faces;

    while (inputVideo.read(frame)) {
        if (frameCount % 2 == 0) {
            faces = pipeline.run(frame);
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
    std::string command = "python ../scripts/sqlite.py \"" + userName +"\" \"" + faces[0].getExpressionText() + "\"";
    system(command.c_str());
}

void FERPipeline::visualize()
{
    std::string command = "python ../scripts/visualize.py \"" + userName +"\"";
    system(command.c_str());
}