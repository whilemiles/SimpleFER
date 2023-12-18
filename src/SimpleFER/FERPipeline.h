#include <iostream>
#include <opencv2/core/types.hpp>
#include "Face.h"

std::vector<Face> FERPipeline(cv::Mat img);

std::vector<Face> detectFace(cv::Mat img);
// cv::Mat alignFace(cv::Mat imgGray);
// cv::Mat normalizeFace(cv::Mat imgGray);
std::vector<Face> analyzeFace(cv::Mat img, std::vector<Face> faces);
