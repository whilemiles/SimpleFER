#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>


int main(int argc, char* argv[])
{
    
    cv::VideoCapture cap;   //声明相机捕获对象

    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G')); 

    int deviceID = 0; //相机设备号
    cap.open(deviceID); //打开相机

    if (!cap.isOpened()) //判断相机是否打开
    {
        std::cerr << "ERROR!!Unable to open camera\n";
        return -1;
    }

    cv::namedWindow("SimpleFER", 1); //创建一个窗口用于显示图像，1代表窗口适应图像的分辨率进行拉伸。
    cv::Mat img;
    while (true)
    {
        cap >> img; //以流形式捕获图像

        if (!img.empty()) //图像不为空则显示图像
        {
            cv::imshow("SimpleFER", img);
        }
        
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