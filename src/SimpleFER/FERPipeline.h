#include <iostream>
#include <opencv2/core/types.hpp>
#include "Face.h"

std::vector<Face> FERPipeline(cv::Mat img);

std::vector<Face> detectFace(cv::Mat img);
std::vector<Face> alignFace(cv::Mat img, std::vector<Face> faces);
std::vector<Face> normalizeFace(cv::Mat img, std::vector<Face> faces);
std::vector<Face> analyzeFace(cv::Mat img, std::vector<Face> faces);
