#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include "FERPipeline.h"
#include "functions.h"

int main(int argc, char* argv[])
{
    cv::VideoCapture cap;   //声明相机捕获对象
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G')); 
    cap.open(0, cv::CAP_V4L2); //打开相机
    if (!cap.isOpened()) //判断相机是否打开
    {
        std::cerr << "ERROR!!Unable to open camera\n";
        return -1;
    }
    
    int interval = 10;
    int frameCount = 0;

    cv::Mat img;
    std::vector<Face> faces;

    cv::namedWindow("SimpleFER", 1); //创建一个窗口用于显示图像，1代表窗口适应图像的分辨率进行拉伸。
    cv::namedWindow("alingedFace", 1);
        
    while (true)
    {
        cap >> img; //以流形式捕获图像

        if (img.empty()){
            break;
        }

        if(frameCount % interval == 0){
            frameCount = 0;
            faces.clear();
            faces = FERPipeline(img);
        }
        for (Face& face : faces)
        {
            cv::rectangle(img, face.region, cv::Scalar(0, 255, 0), 2);
            
            std::string emotion_text = face.getEmotionText();
            cv::Point text_location{face.region.x, face.region.y - 25};
            cv::putText(img, emotion_text, text_location, cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0, 255, 0), 2);
            cv::circle(img, face.left_eye, 3, cv::Scalar_(0, 0, 255), 2);
            cv::circle(img, face.right_eye, 3, cv::Scalar_(255, 0, 0), 2);
        }
        
        if(faces.size() > 0)
        { 
            cv::Mat rotatedImg = img.clone();
            cv::putText(rotatedImg, std::to_string(faces[0].align_angle), 
                    {faces[0].region.x, faces[0].region.y - 50}, 
                    cv::FONT_HERSHEY_PLAIN, 3, cv::Scalar(0, 255, 0), 2);
            rotateImage(rotatedImg, rotatedImg, faces[0].align_angle, 1);
            cv::imshow("alingedFace", rotatedImg);
        }
        
        cv::imshow("SimpleFER", img);

        frameCount++;
        
        int key = cv::waitKey(30); //等待30ms
        if (key ==  int('q')) //按下q退出
        {
            break;
        }
    }
    cap.release(); //释放相机捕获对象
    cv::destroyAllWindows(); //关闭所有窗口

    return 0;
}