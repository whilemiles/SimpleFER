#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include "FERPipeline.h"
#include "functions.h"

using namespace cv;

int main(int argc, char* argv[])
{
    VideoCapture inputVideo("test.mp4");

    if (!inputVideo.isOpened()) {
        std::cerr << "Error: Unable to open input video file\n";
        return -1;
    }

    double fps = inputVideo.get(CAP_PROP_FPS);
    Size frameSize(static_cast<int>(inputVideo.get(CAP_PROP_FRAME_WIDTH)),
                   static_cast<int>(inputVideo.get(CAP_PROP_FRAME_HEIGHT)));
    int totalFrames = static_cast<int>(inputVideo.get(CAP_PROP_FRAME_COUNT));

    VideoWriter outputVideo("output_video.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), fps, frameSize);

    if (!outputVideo.isOpened()) {
        std::cerr << "Error: Unable to create output video file\n";
        return -1;
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
    inputVideo.release();
    outputVideo.release();

    return 0;
}