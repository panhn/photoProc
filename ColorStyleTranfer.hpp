#include <iostream> 
#include <algorithm>
#include <chrono> 
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp> 
#include <omp.h>

static std::string windowname1 = "ColorStyleTranfer";
static cv::Mat src;
static cv::Mat pattern;
static int level = 100;

cv::Mat ShiftColor(cv::Mat src, cv::Mat pat, int n);

void onChangeLevel(int, void*)
{
	auto starttime = std::chrono::system_clock::now();
	cv::Mat res = ShiftColor(src, pattern, level);
	auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - starttime).count();
	std::cout << "所耗时间为：" << diff << "ms" << std::endl;
	cv::imshow(windowname1, res);
	cv::imwrite("res.png", res);
}

cv::Mat ShiftColor(cv::Mat src, cv::Mat pat, int n) {
	cv::Mat labsrc;
	cv::Mat labpat;
	cv::cvtColor(src, labsrc, cv::COLOR_BGR2Lab);
	cv::cvtColor(pattern, labpat, cv::COLOR_BGR2Lab);
	cv::Scalar s_mean, s_stddev;
	cv::Scalar t_mean, t_stddev;
	cv::meanStdDev(labsrc, s_mean, s_stddev);
	cv::meanStdDev(labpat, t_mean, t_stddev);

	float ratio = n / 200.0;
	int height = src.rows;
	int width = src.cols;
	int channel = src.channels();

#pragma omp parallel for
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			for (int k = 0; k < channel; k++) {
				int x = labsrc.at<cv::Vec3b>(i, j)[k];
				//int newVal = x * pow(t_stddev[k] / s_stddev[k], ratio) - (s_mean[k] * t_stddev[k] / s_stddev[k] - t_mean[k]) * ratio;
				int newVal = x * (1 - ratio + t_stddev[k] / s_stddev[k] * ratio) - (s_mean[k] * t_stddev[k] / s_stddev[k] - t_mean[k]) * ratio;
				newVal = std::max(0, std::min(255, newVal));
				labsrc.at<cv::Vec3b>(i, j)[k] = newVal;
			}
		}
	}
	cv::Mat res;
	cv::cvtColor(labsrc, res, cv::COLOR_Lab2BGR);
	return res;
}