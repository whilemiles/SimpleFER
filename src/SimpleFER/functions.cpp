#include "functions.h"

double getEuclideanDistance(int x1 ,int y1 , int x2 , int y2 )
{
	return  pow(pow(x1 - x2, 2) + pow(y1 - y2, 2), 0.5);
}

int rotateImage(const cv::Mat &src, cv::Mat &dst, const double angle, const int mode)
{
	//mode = 0 ,Keep the original image size unchanged
	//mode = 1, Change the original image size to fit the rotated scale, padding with zero

	if (src.empty())
	{
		return -1;
	}

	if (mode == 1)
	{
		cv::Point2f center((src.cols - 1) / 2.0, (src.rows - 1) / 2.0);
		cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
		cv::warpAffine(src, dst, rot, src.size());//the original size
	}
	else {

		double alpha = -angle * CV_PI / 180.0;//convert angle to radian format 

		cv::Point2f srcP[3];
		cv::Point2f dstP[3];
		srcP[0] = cv::Point2f(0, src.rows);
		srcP[1] = cv::Point2f(src.cols, 0);
		srcP[2] = cv::Point2f(src.cols, src.rows);
		
		//rotate the pixels
		for (int i=0;i<3;i++)
					dstP[i] = cv::Point2f(srcP[i].x*cos(alpha) - srcP[i].y*sin(alpha), srcP[i].y*cos(alpha) + srcP[i].x*sin(alpha));
		double minx, miny, maxx, maxy;
		minx = std::min(std::min(std::min(dstP[0].x, dstP[1].x), dstP[2].x),float(0.0));
		miny  = std::min(std::min(std::min(dstP[0].y, dstP[1].y), dstP[2].y),float(0.0));
		maxx = std::max(std::max(std::max(dstP[0].x, dstP[1].x), dstP[2].x),float(0.0));
		maxy = std::max(std::max(std::max(dstP[0].y, dstP[1].y), dstP[2].y),float(0.0));

		int w = maxx - minx;
		int h = maxy - miny;

		//translation
		for (int i = 0; i < 3; i++)
		{
			if (minx < 0)
				dstP[i].x -= minx;
			if (miny < 0)
				dstP[i].y -= miny;
		}

		cv::Mat warpMat = cv::getAffineTransform(srcP, dstP);
		cv::warpAffine(src, dst, warpMat, cv::Size(w, h));//extend size

	}//end else

	return 0;
}
