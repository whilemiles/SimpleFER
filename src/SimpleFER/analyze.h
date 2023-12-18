#include <iostream>
#include <opencv2/core/types.hpp>
#include "Face.h"

std::vector<Face> analyzeFace(cv::Mat img);
cv::Mat alignImage(cv::Mat imgGray);
