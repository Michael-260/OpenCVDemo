#include <iostream>
#include<fstream>
#include<math.h>

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

const int slidermax = 128;
const int anglemax = 90;

	void operators_demo(Mat& image);//提高亮度

	Mat RGB_to_HSI(Mat&image);//BGRtoHSI色彩转换
	Mat HSI_to_RGB(Mat&image);//HSItoBGR色彩转换

	void pixel_read(Mat&image);//通过指针方式访问像素点

	void saving(Mat&image);


	const String origin_image = "原图像";
	const String result_image = "显示结果";


void ColorReduce(Mat& input, Mat& output, int div);
void on_trackbar(int pos, void* userdata);

void test01();//滑杆控件
void test02();//RGB转HSI
void test03();//HSI转RGB

void test04();//灰度直方图的统计
void test06();//实现均值滤波、高斯平滑
void test07();//图像锐化

void test05();//图像几何变换—旋转平移

void test08();//实现傅里叶变换

void test09();//迭代阈值法
void test10();//区域生长法
void test11();//Canny算法
void test12();//哈夫变换(识别直线和圆)
void test13();//角点检测









//Mat RGB_TO_HSI(Mat src) {
//	int row = src.rows;
//	int col = src.cols;
//	Mat dsthsi(row, col, CV_64FC3);
//	/*Mat H = Mat(row, col, CV_64FC1);
//	Mat S = Mat(row, col, CV_64FC1);
//	Mat I = Mat(row, col, CV_64FC1);*/
//	for (int i = 0; i < row; i++) {
//		for (int j = 0; j < col; j++) {
//			double h, s, newi, th;
//			double B = (double)src.at<Vec3b>(i, j)[0] / 255.0;
//			double G = (double)src.at<Vec3b>(i, j)[1] / 255.0;
//			double R = (double)src.at<Vec3b>(i, j)[2] / 255.0;
//			double mi, mx;
//			if (R > G && R > B) {
//				mx = R;
//				mi = min(G, B);
//			}
//			else {
//				if (G > B) {
//					mx = G;
//					mi = min(R, B);
//				}
//				else {
//					mx = B;
//					mi = min(R, G);
//				}
//			}
//			newi = (R + G + B) / 3.0;
//			if (newi < 0)  newi = 0;
//			else if (newi > 1) newi = 1.0;
//			if (newi == 0 || mx == mi) {
//				s = 0;
//				h = 0;
//			}
//			else {
//				s = 1 - mi / newi;
//				th = (R - G) * (R - G) + (R - B) * (G - B);
//				th = sqrt(th) /*+ 1e-5*/;
//				th = acos(((R - G + R - B) * 0.5) / th);
//				if (G >= B) h = th;
//				else h = 2 * CV_PI - th;
//			}
//			h = h / (2 * CV_PI);
//			/*H.at<double>(i, j) = h;
//			S.at<double>(i, j) = s;
//			I.at<double>(i, j) = newi;*/
//
//			dsthsi.at<Vec3d>(i, j)[0] = h;
//			dsthsi.at<Vec3d>(i, j)[1] = s;
//			dsthsi.at<Vec3d>(i, j)[2] = newi;
//
//		}
//	}
//	return dsthsi;
//}

/*void test(int i[5]) {
	int* p = i;
	*p++ = *p+1;
	*p++ = *p + 1;
	*p++ = *p + 1;
	*p++ = *p + 1;
	for (int j = 0; j < 5;j++)cout << i[j];
}*///通过指针访问像素点

