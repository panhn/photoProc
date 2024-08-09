#include <iostream> 
#include "func.hpp"

using namespace cv;
using namespace std;

int main()
{
	input = imread("D:/test_img/TP (476).JPG");

	namedWindow(windowname, 0);
	resizeWindow(windowname, 540, 900);
	imshow(windowname, input);
	
	/*auto starttime = chrono::system_clock::now();

	cv::Mat out = ColorTemperature1(src, temp);

	auto diff = chrono::duration_cast<std::chrono::milliseconds>(chrono::system_clock::now() - starttime).count();
	cout << "所耗时间为：" << diff << "ms" << endl;
	cv::imwrite("out1.jpg", out);*/


	//createTrackbar("tempera", windowname, &temperature, 2 * maxValue, onChangeTempera);
	//createTrackbar("tempera1", windowname, &temperature1, 2 * maxValue, onChangeTempera1);
	//createTrackbar("shadow", windowname, &light, 2 * maxValue, onChangeShadow);
	//createTrackbar("saturation", windowname, &sat, 2 * maxValue, onChangeSaturation);
	//createTrackbar("definition", windowname, &def, 2 * maxValue, onChangeDefinition);
	//createTrackbar("exposure", windowname, &tone, 2 * maxValue, onChangeExposure);
	//createTrackbar("vividness", windowname, &vividness, 2 * maxValue, onChangeVividness);
	
	waitKey();
	return 0;
}