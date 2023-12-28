#include <iostream>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include "Face.h"

class FERPipeline
{
    cv::Mat inputImage;
    std::vector<Face> faces;
public:
    std::vector<Face> run(cv::Mat img);
private:
    void detect();
    void align();
    void normalize();
    void analyze();
};