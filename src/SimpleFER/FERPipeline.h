#include <iostream>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/objdetect/face.hpp>
#include "Face.h"

class FERPipeline
{
    std::string userName;
    cv::Mat inputImage;
    cv::Mat alignedImage;
    std::vector<Face> faces;
    cv::Ptr<cv::FaceDetectorYN> detector;
    cv::Ptr<cv::FaceRecognizerSF> faceRecognizer;
public:
    FERPipeline(){
        detector = cv::FaceDetectorYN::create("../saved/face_detection_yunet_2023mar.onnx", "", {});
        faceRecognizer = cv::FaceRecognizerSF::create("../saved/face_recognition_sface_2021dec.onnx", "");
        userName = "user";
    }
    std::vector<Face> run(cv::Mat img);
    void offline_process(std::string filename);
    void save();
    void visualize();
private:
    void detect();
    void align();
    void normalize();
    void analyze();
};