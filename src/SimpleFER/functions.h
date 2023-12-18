#include <cmath>
#include <algorithm>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

double getEuclideanDistance(int x1 ,int y1 , int x2 , int y2 );
int rotateImage(const cv::Mat &src, cv::Mat &dst, const double angle, const int mode);



