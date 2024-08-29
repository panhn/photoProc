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
Mat HistogramMatching(const Mat& source, const Mat& template_img);

void onChangeLevel(int, void*)
{
	auto starttime = std::chrono::system_clock::now();
	cv::Mat res = ShiftColor(src, pattern, level);
	auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - starttime).count();
	std::cout << "所耗时间为：" << diff << "ms" << std::endl;
	cv::imshow(windowname1, res);
	cv::imwrite("res.png", res);
}

// 使用均值与方差来迁移色彩风格
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

// --------------------------------------------
// --------------------------------------------
//基于直方图的色彩迁移
cv::Mat colorTransfer(cv::Mat sourceImage, cv::Mat patternImage) {
    // 分离源图像和目标图像的通道
    std::vector<cv::Mat> sourceChannels;
    cv::split(sourceImage, sourceChannels);

    std::vector<cv::Mat> patternChannels;
    cv::split(patternImage, patternChannels);

    // 对每个通道进行直方图匹配
    for (int i = 0; i < 3; i++) {
        sourceChannels[i] = HistogramMatching(sourceChannels[i], patternChannels[i]);
    }

    // 合并匹配后的通道
    cv::Mat transferredImage;
    cv::merge(sourceChannels, transferredImage);
    
    cv::namedWindow("transferredImage", 0);
    resizeWindow("transferredImage", 540, 900);
    cv::imshow("transferredImage", transferredImage);
    imwrite("res.png", transferredImage);
    return transferredImage;
}

Mat HistogramMatching(const Mat& source, const Mat& template_img) {
    // 计算源图像和模板图像的直方图
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    cv:: Mat src_hist, tmpl_hist;

    calcHist(&source, 1, 0, Mat(), src_hist, 1, &histSize, &histRange);
    calcHist(&template_img, 1, 0, Mat(), tmpl_hist, 1, &histSize, &histRange);

    // 计算累积分布函数（CDF）
    src_hist /= source.total();
    tmpl_hist /= template_img.total();

    vector<float> cdf_src(histSize, 0);
    vector<float> cdf_tmpl(histSize, 0);

    cdf_src[0] = src_hist.at<float>(0);
    cdf_tmpl[0] = tmpl_hist.at<float>(0);

    for (int i = 1; i < histSize; i++) {
        cdf_src[i] = cdf_src[i - 1] + src_hist.at<float>(i);
        cdf_tmpl[i] = cdf_tmpl[i - 1] + tmpl_hist.at<float>(i);
    }

    // 创建映射表
    vector<int> lut(histSize, 0);
    int tmpl_idx = 0;

    for (int src_idx = 0; src_idx < histSize; src_idx++) {
        while (tmpl_idx < histSize && cdf_tmpl[tmpl_idx] < cdf_src[src_idx]) {
            tmpl_idx++;
        }
        lut[src_idx] = tmpl_idx;
    }

    // 应用映射表
    Mat matched = source.clone();
    for (int y = 0; y < source.rows; y++) {
        for (int x = 0; x < source.cols; x++) {
            matched.at<uchar>(y, x) = saturate_cast<uchar>(lut[source.at<uchar>(y, x)]);
        }
    }

    return matched;
}
