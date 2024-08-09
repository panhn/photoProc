#include <iostream> 
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp> 

#include <chrono>   //����ʱ��

using namespace cv;
using namespace std;

static string windowname = "WhiteBalance";
static Mat input;
static int temperature = 100;
static int temperature1 = 100;
static int tone = 100;
static int light = 100;
static int sat = 100;
static int def = 100;
static int vividness = 100;
static int maxValue = 100;

cv::Mat ColorTemperature(cv::Mat src, int n);
cv::Mat ColorTemperature1(cv::Mat src, int n);
cv::Mat ChangeTone(cv::Mat image, int n);
cv::Mat ChangeShadow(cv::Mat src, int n);
cv::Mat ChangeShadow1(cv::Mat src, int n);
cv::Mat ChangeSaturation(cv::Mat src, int n);
cv::Mat ChangeDefinition(cv::Mat src, int n);
cv::Mat ChangeExposure(cv::Mat src, int n);
cv::Mat ChangeVividness(cv::Mat src, int n);

void onChangeTempera(int, void*)
{
	Mat dst = ColorTemperature(input, temperature - 100);
	imshow(windowname, dst);
	cv::imwrite("out.jpg", dst);
}

void onChangeTempera1(int, void*)
{
	auto starttime = chrono::system_clock::now();

	Mat dst = ColorTemperature1(input, temperature1 - 100);

	auto diff = chrono::duration_cast<std::chrono::milliseconds>(chrono::system_clock::now() - starttime).count();
	cout << "����ʱ��Ϊ��" << diff << "ms" << endl;
	imshow(windowname, dst);
	cv::imwrite("out1.jpg", dst);
}

void onChangeTone(int, void*) {
	Mat dst = ChangeTone(input, tone - 100);
	imshow(windowname, dst);
	cv::imwrite("out1.jpg", dst);
}

void onChangeShadow(int, void*)
{
	Mat dst = ChangeShadow1(input, light - 100);
	imshow(windowname, dst);
	cv::imwrite("out1.jpg", dst);
}

void onChangeSaturation(int, void*)
{
	Mat dst = ChangeSaturation(input, sat - 100);
	imshow(windowname, dst);
	cv::imwrite("out1.jpg", dst);
}

void onChangeDefinition(int, void*) {
	cout << "------------------- test change definition --------------------" << endl;
	cv::Mat dst = ChangeDefinition(input, def - 100);
	imshow(windowname, dst);
	cv::imwrite("out1.jpg", dst);
}

void onChangeExposure(int, void*) {
	Mat dst = ChangeExposure(input, tone - 100);
	imshow(windowname, dst);
	cv::imwrite("out1.jpg", dst);
}

void onChangeVividness(int, void*) {
	Mat dst = ChangeVividness(input, vividness - 100);
	imshow(windowname, dst);
	cv::imwrite("out1.jpg", dst);
}

//--------------------------------------------------------------------------------
// ��ƽ��-��������
cv::Mat WhiteBalcane_PRA(cv::Mat src)
{
	cv::Mat result = src.clone();
	if (src.channels() != 3)
	{
		std::cout << "The number of image channels is not 3." << std::endl;
		return result;
	}

	// ͨ������
	std::vector<cv::Mat> Channel;
	cv::split(src, Channel);

	// �������
	int row = src.rows;
	int col = src.cols;
	int RGBSum[766] = { 0 };
	uchar maxR, maxG, maxB;

	// ���㵥ͨ�����ֵ
	for (int i = 0; i < row; ++i)
	{
		uchar* b = Channel[0].ptr<uchar>(i);
		uchar* g = Channel[1].ptr<uchar>(i);
		uchar* r = Channel[2].ptr<uchar>(i);
		for (int j = 0; j < col; ++j)
		{
			int sum = b[j] + g[j] + r[j];
			RGBSum[sum]++;
			maxB = cv::max(maxB, b[j]);
			maxG = cv::max(maxG, g[j]);
			maxR = cv::max(maxR, r[j]);
		}
	}

	// ����������������T
	int T = 0;
	int num = 0;
	int K = static_cast<int>(row * col * 0.1);
	for (int i = 765; i >= 0; --i)
	{
		num += RGBSum[i];
		if (num > K)
		{
			T = i;
			break;
		}
	}

	// ���㵥ͨ������ƽ��ֵ
	double Bm = 0.0, Gm = 0.0, Rm = 0.0;
	int count = 0;
	for (int i = 0; i < row; ++i)
	{
		uchar* b = Channel[0].ptr<uchar>(i);
		uchar* g = Channel[1].ptr<uchar>(i);
		uchar* r = Channel[2].ptr<uchar>(i);
		for (int j = 0; j < col; ++j)
		{
			int sum = b[j] + g[j] + r[j];
			if (sum > T)
			{
				Bm += b[j];
				Gm += g[j];
				Rm += r[j];
				count++;
			}
		}
	}
	Bm /= count;
	Gm /= count;
	Rm /= count;

	// ͨ������
	Channel[0] *= maxB / Bm;
	Channel[1] *= maxG / Gm;
	Channel[2] *= maxR / Rm;

	// �ϲ�ͨ��
	cv::merge(Channel, result);

	return result;
}

// ��ƽ��-�Ҷ�����
cv::Mat gratWorld_WhiteBalance(cv::Mat src) {
	cv::Mat result = src.clone();
	vector<Mat> imageRGB;

	//RGB��ͨ������
	split(src, imageRGB);

	//��ԭʼͼ���RGB�����ľ�ֵ
	double R, G, B;
	B = mean(imageRGB[0])[0];
	G = mean(imageRGB[1])[0];
	R = mean(imageRGB[2])[0];

	//��Ҫ������RGB����������
	double KR, KG, KB;
	KB = (R + G + B) / (3 * B);
	KG = (R + G + B) / (3 * G);
	KR = (R + G + B) / (3 * R);

	//����RGB����ͨ�����Ե�ֵ
	imageRGB[0] = imageRGB[0] * KB;
	imageRGB[1] = imageRGB[1] * KG;
	imageRGB[2] = imageRGB[2] * KR;

	//RGB��ͨ��ͼ��ϲ�
	merge(imageRGB, result);

	return result;
}

//--------------------------------------------------------------------------------
// ɫ�µ��ڷ�ʽһ
void GetRGBfromTemperature(int& r, int& g, int& b, int tmpKelvin) {
	double tmpCalc;

	// ȷ���¶�����Ч��Χ��
	if (tmpKelvin < 1000) tmpKelvin = 1000;
	if (tmpKelvin > 40000) tmpKelvin = 40000;

	// ���¶ȳ��� 100
	tmpKelvin /= 100;

	// �����ɫ
	if (tmpKelvin <= 66) {
		r = 255;
	}
	else {
		tmpCalc = tmpKelvin - 55;
		r = 351.976905668057 + 0.114206453784165 * tmpCalc - 40.2536630933213 * log(tmpCalc);
		if (r < 0) r = 0;
		if (r > 255) r = 255;
	}

	// ������ɫ
	if (tmpKelvin <= 66) {
		tmpCalc = tmpKelvin - 2;
		g = -155.254855627092 - 0.445969504695791 * tmpCalc + 104.492161993939 * log(tmpCalc);
		if (g < 0) g = 0;
		if (g > 255) g = 255;
	}
	else {
		tmpCalc = tmpKelvin - 50;
		g = 325.449412571197 + 0.0794345653666234 * tmpCalc - 28.0852963507957 * log(tmpCalc);
		if (g < 0) g = 0;
		if (g > 255) g = 255;
	}

	// ������ɫ
	if (tmpKelvin >= 66) {
		b = 255;
	}
	else if (tmpKelvin <= 19) {
		b = 0;
	}
	else {
		tmpCalc = tmpKelvin - 10;
		b = -254.769351841209 + 0.827409606400739 * tmpCalc + 115.679944010661 * log(tmpCalc);
		if (b < 0) b = 0;
		if (b > 255) b = 255;
	}
}

double GetLuminance(int r, int g, int b)
{
	double R, G, B, Max, Min;
	R = r / 255.0;       //Where RGB values = 0 �� 255
	G = g / 255.0;
	B = b / 255.0;

	Min = min(R, min(G, B));    //Min. value of RGB
	Max = max(R, max(G, B));    //Max. value of RGB

	return (Max + Min) / 2.0;
}

void RGB2HSL(int r, int g, int b, double& H, double& S, double& L)
{
	double R, G, B, Max, Min, del_R, del_G, del_B, del_Max;
	R = r / 255.0;
	G = g / 255.0;
	B = b / 255.0;

	Min = min(R, min(G, B));    //Min. value of RGB
	Max = max(R, max(G, B));    //Max. value of RGB
	del_Max = Max - Min;        //Delta RGB value

	L = (Max + Min) / 2.0;

	if (del_Max == 0)           //This is a gray, no chroma...
	{
		//H = 2.0/3.0;          //Windows��SֵΪ0ʱ��Hֵʼ��Ϊ160��2/3*240��
		H = 0;                  //HSL results = 0 �� 1
		S = 0;
	}
	else                        //Chromatic data...
	{
		if (L < 0.5) S = del_Max / (Max + Min);
		else         S = del_Max / (2 - Max - Min);

		del_R = (((Max - R) / 6.0) + (del_Max / 2.0)) / del_Max;
		del_G = (((Max - G) / 6.0) + (del_Max / 2.0)) / del_Max;
		del_B = (((Max - B) / 6.0) + (del_Max / 2.0)) / del_Max;

		if (R == Max) H = del_B - del_G;
		else if (G == Max) H = (1.0 / 3.0) + del_R - del_B;
		else if (B == Max) H = (2.0 / 3.0) + del_G - del_R;

		if (H < 0)  H += 1;
		if (H > 1)  H -= 1;
	}
}

double Hue2RGB(double v1, double v2, double vH)
{
	if (vH < 0) vH += 1;
	if (vH > 1) vH -= 1;
	if (6.0 * vH < 1) return v1 + (v2 - v1) * 6.0 * vH;
	if (2.0 * vH < 1) return v2;
	if (3.0 * vH < 2) return v1 + (v2 - v1) * ((2.0 / 3.0) - vH) * 6.0;
	return (v1);
}

void HSL2RGB(double H, double S, double L, int& R, int& G, int& B)
{
	double var_1, var_2;
	if (S == 0)                       //HSL values = 0 �� 1
	{
		R = L * 255.0;                //RGB results = 0 �� 255
		G = L * 255.0;
		B = L * 255.0;
	}
	else
	{
		if (L < 0.5) var_2 = L * (1 + S);
		else         var_2 = (L + S) - (S * L);

		var_1 = 2.0 * L - var_2;

		R = 255.0 * Hue2RGB(var_1, var_2, H + (1.0 / 3.0));
		G = 255.0 * Hue2RGB(var_1, var_2, H);
		B = 255.0 * Hue2RGB(var_1, var_2, H - (1.0 / 3.0));
	}
}

int BlendColors(int color, int tmp, double tempStrength) {
	return (1 - tempStrength) * color + tmp * tempStrength;
}

cv::Mat ColorTemperature(cv::Mat src, int n)
{
	Mat dst = src.clone();
	
	int newTemperature;  // ���¶�ֵ
	if (n < 0) {
		newTemperature = 6600 + n / 100.0 * 5600;
	}
	else {
		newTemperature = 6600 + n / 100.0 * 33400;
	}
	bool preserveLuminance = true;  // ����Ĭ���Ƿ�������
	double tempStrength = 0.25;  // ����Ĭ�ϵ���ǿ��

	// ��ɫ����
	int r = 0, g = 0, b = 0;
	double h = 0.0, s = 0.0, l = 0.0;
	double originalLuminance = 0.0;
	int tmpR = 0, tmpG = 0, tmpB = 0;

	// ��ȡ��Ӧ�¶ȵ� RGB ֵ����Ҫ����ʵ�֣�
	GetRGBfromTemperature(tmpR, tmpG, tmpB, newTemperature);

	// ����ͼ������
	for (int y = 0; y < src.rows; ++y) {
		for (int x = 0; x < src.cols; ++x) {
			Vec3b color = src.at<Vec3b>(y, x);
			// ��ȡԴ������ɫֵ
			b = color[0];
			g = color[1];
			r = color[2];

			// ���Ҫ�������ȣ������ʼ����ֵ
			originalLuminance = GetLuminance(r, g, b);

			// ��ɫ���
			r = BlendColors(r, tmpR, tempStrength);
			g = BlendColors(g, tmpG, tempStrength);
			b = BlendColors(b, tmpB, tempStrength);

			b = min(max(b, 0), 255);
			g = min(max(g, 0), 255);
			r = min(max(r, 0), 255);

			// ���Ҫ�������ȣ����¼�����ɫֵ
			RGB2HSL(r, g, b, h, s, l);
			HSL2RGB(h, s, originalLuminance, r, g, b);

			// ��ֵ����ɫֵ
			dst.at<Vec3b>(y, x) = Vec3b(b, g, r);
		}
	}

	return dst;
}

// ɫ�µ��ڷ�ʽ��
cv::Mat ColorTemperature1(cv::Mat image, int n) {
	Mat result = image.clone();

	auto clamp = [](int value, int min, int max) ->int {
		if (value < min) return min;
		else if (value > max) return max;
		return value;
		};

	int blue = 0, green = 0, red = 0;

	// ����������ɫͨ��
	float beta = n / 100.0;
	for (int y = 0; y < image.rows; ++y) {
		for (int x = 0; x < image.cols; ++x) {
			Vec3b color = image.at<Vec3b>(y, x);

			blue = color[0];
			green = color[1];
			red = color[2];

			if (beta >= 0) {
				blue = blue + (float)(blue - 255) * blue * beta * 0.13 / 255;
				green = green + (float)green * (255 - green) * beta * 0.8 / 255;
				red = red + (float)red * (255 - red) * beta * 2.6 / 255;
			}
			else {
				blue = blue + (float)(blue - 255) * blue * beta * 4 / 255;
				green = green + (float)green * (green - 255) * beta * 0.6 / 255;
				red = red + (float)red * (255 - red) * beta * 0.4 / 255;;
			}

			blue = clamp(blue, 0, 255);
			green = clamp(green, 0, 255);
			red = clamp(red, 0, 255);

			result.at<Vec3b>(y, x) = Vec3b(blue, green, red);
		}
	}
	return result;
}

// ɫ������
cv::Mat ChangeTone(cv::Mat image, int n) {
	Mat result = image.clone();

	auto clamp = [](int value, int min, int max) ->int {
		if (value < min) return min;
		else if (value > max) return max;
		return value;
	};

	// ����������ɫͨ��
	float beta = n / 100.0;
	for (int y = 0; y < image.rows; ++y) {
		for (int x = 0; x < image.cols; ++x) {
			Vec3b color = image.at<Vec3b>(y, x);

			int blue = color[0];
			int green = color[1];
			int red = color[2];

			if (beta >= 0) {
				blue = blue + (float)(255 - blue) * blue * beta * beta * 1.3 / 255;
				green = green + (float)green * (green - 255) * beta * 0.1 / 255;
				red = red + (float)red * (255 - red) * beta * 0.5 / 255;
			}
			else {
				blue = blue + (float)(255 - blue) * blue * beta * 0.2 / 255;
				green = green + (float)green * (green - 255) * beta * 1.1 / 255;
				red = red + (float)red * (red - 255) * beta * 0.5 / 255;;
			}

			blue = clamp(blue, 0, 255);
			green = clamp(green, 0, 255);
			red = clamp(red, 0, 255);

			result.at<Vec3b>(y, x) = Vec3b(blue, green, red);
		}
	}
	return result;
}

//--------------------------------------------------------------------------------
// �ع�
cv::Mat ChangeExposure(cv::Mat src, int n) {
	float exposureFactor = (n + 100) / 100.0;
	cv::Mat dst;
	src.convertTo(dst, -1, exposureFactor, 0);
	return dst;
}

//--------------------------------------------------------------------------------
// �����ȵ���
cv::Mat ChangeDefinition(cv::Mat src, int n) {
	float amount = n / 100.0;
	cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
		0, -1, 0,
		-1, 5 + amount, -1,
		0, -1, 0);
	cv::Mat dst;
	cv::filter2D(src, dst, src.depth(), kernel);
	return dst;
}

//--------------------------------------------------------------------------------
// ��Ӱ����
cv::Mat ChangeShadow(cv::Mat src, int shadow)
{
	// ���ɻҶ�ͼ
	cv::Mat gray = cv::Mat::zeros(src.size(), CV_32FC1);
	cv::Mat f = src.clone();
	f.convertTo(f, CV_32FC3);
	std::vector<cv::Mat> pics;
	split(f, pics);
	gray = 0.299f * pics[2] + 0.587 * pics[2] + 0.114 * pics[0];
	gray = gray / 255.f;

	// ȷ����Ӱ��
	cv::Mat thresh = cv::Mat::zeros(gray.size(), gray.type());
	thresh = (1.0f - gray).mul(1.0f - gray);
	// ȡƽ��ֵ��Ϊ��ֵ
	cv::Scalar t = mean(thresh);
	cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8UC1);
	mask.setTo(255, thresh >= t[0]);

	// ��������
	int max = 4;
	float bright = shadow / 400.0f / max;
	float mid = 1.0f + max * bright;

	// ��Եƽ������
	cv::Mat midrate = cv::Mat::zeros(src.size(), CV_32FC1);
	cv::Mat brightrate = cv::Mat::zeros(src.size(), CV_32FC1);
	for (int i = 0; i < src.rows; ++i)
	{
		uchar* m = mask.ptr<uchar>(i);
		float* th = thresh.ptr<float>(i);
		float* mi = midrate.ptr<float>(i);
		float* br = brightrate.ptr<float>(i);
		for (int j = 0; j < src.cols; ++j)
		{
			if (m[j] == 255)
			{
				mi[j] = mid;
				br[j] = bright;
			}
			else {
				mi[j] = (mid - 1.0f) / t[0] * th[j] + 1.0f;
				br[j] = (1.0f / t[0] * th[j]) * bright;
			}
		}
	}

	// ��Ӱ��������ȡ���ͼ
	cv::Mat result = cv::Mat::zeros(src.size(), src.type());
	for (int i = 0; i < src.rows; ++i)
	{
		float* mi = midrate.ptr<float>(i);
		float* br = brightrate.ptr<float>(i);
		uchar* in = src.ptr<uchar>(i);
		uchar* r = result.ptr<uchar>(i);
		for (int j = 0; j < src.cols; ++j)
		{
			for (int k = 0; k < 3; ++k)
			{
				float temp = pow(float(in[3 * j + k]) / 255.f, 1.0f / mi[j]) * (1.0 / (1 - br[j]));
				if (temp > 1.0f)
					temp = 1.0f;
				if (temp < 0.0f)
					temp = 0.0f;
				uchar utemp = uchar(255 * temp);
				r[3 * j + k] = utemp;
			}

		}
	}
	return result;
}

cv::Mat ChangeShadow1(cv::Mat src, int shadow) {
	float factor = shadow / 200.0;

	cv::Mat result;
	cv::cvtColor(src, result, cv::COLOR_BGR2HSV);

	for (int i = 0; i < result.rows; ++i) {
		for (int j = 0; j < result.cols; ++j) {
			uchar value = result.at<cv::Vec3b>(i, j)[2];
			double adjustmentFactor = (255 - value) / 255.0 * factor;  // �������ȼ����������
			result.at<cv::Vec3b>(i, j)[2] = cv::saturate_cast<uchar>(value * (1 + adjustmentFactor));
		}
	}

	cv::cvtColor(result, result, cv::COLOR_HSV2BGR);
	return result;
}

//--------------------------------------------------------------------------------
// �ı�����
cv::Mat ChangeBright(cv::Mat src, int n) {
	cv::Mat dst;
	dst = cv::Mat::zeros(src.size(), src.type());		//�½��հ�ģ�壺��С/������ԭͼ��һ�£�����ֵȫ0��
	int height = src.rows;								//��ȡͼ��߶�
	int width = src.cols;								//��ȡͼ����
	float alpha = (float)n / 100;						//���ȣ�0~1Ϊ����1~������Ϊ����
	float beta = 0.0;									//�Աȶ�

	cv::Mat template1;
	src.convertTo(template1, CV_32F);					//��CV_8UC1ת��ΪCV32F1���ݸ�ʽ��
	for (int row = 0; row < height; row++)
	{
		for (int col = 0; col < width; col++)
		{
			if (src.channels() == 3)
			{
				float b = template1.at<cv::Vec3f>(row, col)[0];		//��ȡͨ��������ֵ��blue��
				float g = template1.at<cv::Vec3f>(row, col)[1];		//��ȡͨ��������ֵ��green��
				float r = template1.at<cv::Vec3f>(row, col)[2];		//��ȡͨ��������ֵ��red��

				//cv::saturate_cast<uchar>(vaule)����ע�⣬valueֵ��Χ������0~255֮�䡣
				dst.at<cv::Vec3b>(row, col)[0] = cv::saturate_cast<uchar>(b * alpha + beta);		//�޸�ͨ��������ֵ��blue��
				dst.at<cv::Vec3b>(row, col)[1] = cv::saturate_cast<uchar>(g * alpha + beta);		//�޸�ͨ��������ֵ��green��
				dst.at<cv::Vec3b>(row, col)[2] = cv::saturate_cast<uchar>(r * alpha + beta);		//�޸�ͨ��������ֵ��red��
			}
			else if (src.channels() == 1)
			{
				float v = src.at<uchar>(row, col);											//��ȡͨ��������ֵ������
				dst.at<uchar>(row, col) = cv::saturate_cast<uchar>(v * alpha + beta);		//�޸�ͨ��������ֵ������
				//saturate_cast<uchar>����Ҫ��Ϊ�˷�ֹ��ɫ������������color<0����color����0�����color>255����color����255��
			}
		}
	}
	return dst;
}

//--------------------------------------------------------------------------------
// �����߹�
cv::Mat ChangeHighLight(cv::Mat src, int highlight)
{
	// ���ɻҶ�ͼ
	cv::Mat gray = cv::Mat::zeros(src.size(), CV_32FC1);
	cv::Mat f = src.clone();
	f.convertTo(f, CV_32FC3);
	std::vector<cv::Mat> pics;
	split(f, pics);
	gray = 0.299f * pics[2] + 0.587 * pics[2] + 0.114 * pics[0];
	gray = gray / 255.f;

	// ȷ���߹���
	cv::Mat thresh = cv::Mat::zeros(gray.size(), gray.type());
	thresh = gray.mul(gray);
	// ȡƽ��ֵ��Ϊ��ֵ
	cv::Scalar t = mean(thresh);
	cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8UC1);
	mask.setTo(255, thresh >= t[0]);

	// ��������
	int max = 4;
	float bright = highlight / 100.0f / max;
	float mid = 1.0f + max * bright;

	// ��Եƽ������
	cv::Mat midrate = cv::Mat::zeros(src.size(), CV_32FC1);
	cv::Mat brightrate = cv::Mat::zeros(src.size(), CV_32FC1);
	for (int i = 0; i < src.rows; ++i)
	{
		uchar* m = mask.ptr<uchar>(i);
		float* th = thresh.ptr<float>(i);
		float* mi = midrate.ptr<float>(i);
		float* br = brightrate.ptr<float>(i);
		for (int j = 0; j < src.cols; ++j)
		{
			if (m[j] == 255)
			{
				mi[j] = mid;
				br[j] = bright;
			}
			else {
				mi[j] = (mid - 1.0f) / t[0] * th[j] + 1.0f;
				br[j] = (1.0f / t[0] * th[j]) * bright;
			}
		}
	}

	// �߹���������ȡ���ͼ
	cv::Mat result = cv::Mat::zeros(src.size(), src.type());
	for (int i = 0; i < src.rows; ++i)
	{
		float* mi = midrate.ptr<float>(i);
		float* br = brightrate.ptr<float>(i);
		uchar* in = src.ptr<uchar>(i);
		uchar* r = result.ptr<uchar>(i);
		for (int j = 0; j < src.cols; ++j)
		{
			for (int k = 0; k < 3; ++k)
			{
				float temp = pow(float(in[3 * j + k]) / 255.f, 1.0f / mi[j]) * (1.0 / (1 - br[j]));
				if (temp > 1.0f)
					temp = 1.0f;
				if (temp < 0.0f)
					temp = 0.0f;
				uchar utemp = uchar(255 * temp);
				r[3 * j + k] = utemp;
			}

		}
	}
	return result;
}

//--------------------------------------------------------------------------------
// ��Ȼ���Ͷ�
cv::Mat ChangeVividness(cv::Mat src, int n) {
	Mat result = src.clone();

	auto clamp = [](int value, int min, int max) ->int {
		if (value < min) return min;
		else if (value > max) return max;
		return value;
		};

	// ����������ɫͨ��
	float beta = n / 100.0;
	for (int y = 0; y < src.rows; ++y) {
		for (int x = 0; x < src.cols; ++x) {
			Vec3b color = src.at<Vec3b>(y, x);

			int blue = color[0];
			int green = color[1];
			int red = color[2];

			float avg = (blue + green + red) / 3.0;
			int maxVal = max(blue, max(green, red));

			int amtVal = (static_cast<float>(maxVal - avg)) / 128 * beta;

			blue += (maxVal - blue) * amtVal;
			green += (maxVal - green) * amtVal;
			red += (maxVal - red) * amtVal;
			
			blue = clamp(blue, 0, 255);
			green = clamp(green, 0, 255);
			red = clamp(red, 0, 255);

			result.at<Vec3b>(y, x) = Vec3b(blue, green, red);
		}
	}
	return result;
}

//--------------------------------------------------------------------------------
// ���Ͷ�
cv::Mat ChangeSaturation(cv::Mat src, int saturation)
{
	float Increment = saturation * 1.0f / 120;
	cv::Mat result = src.clone();
	int row = src.rows;
	int col = src.cols;
	for (int i = 0; i < row; ++i)
	{
		uchar* t = result.ptr<uchar>(i);
		uchar* s = src.ptr<uchar>(i);
		for (int j = 0; j < col; ++j)
		{
			uchar b = s[3 * j];
			uchar g = s[3 * j + 1];
			uchar r = s[3 * j + 2];
			float maxValue = max(r, max(g, b));
			float minValue = min(r, min(g, b));
			float delta, value;
			float L, S, alpha;
			delta = (maxValue - minValue) / 255;
			if (delta == 0)
				continue;
			value = (maxValue + minValue) / 255;
			L = value / 2;
			if (L < 0.5)
				S = delta / value;
			else
				S = delta / (2 - value);
			if (Increment >= 0)
			{
				if ((Increment + S) >= 1)
					alpha = S;
				else
					alpha = 1 - Increment;
				alpha = (1 / alpha - 1) * 0.6;
				t[3 * j + 2] = static_cast<uchar>(r + (r - L * 255) * alpha);
				t[3 * j + 1] = static_cast<uchar>(g + (g - L * 255) * alpha);
				t[3 * j] = static_cast<uchar>(b + (b - L * 255) * alpha);
			}
			else
			{
				alpha = Increment;
				t[3 * j + 2] = static_cast<uchar>(L * 255 + (r - L * 255) * (1 + alpha));
				t[3 * j + 1] = static_cast<uchar>(L * 255 + (g - L * 255) * (1 + alpha));
				t[3 * j] = static_cast<uchar>(L * 255 + (b - L * 255) * (1 + alpha));
			}
		}
	}
	return result;
}