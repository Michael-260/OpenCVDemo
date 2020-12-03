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

	void operators_demo(Mat& image);//�������

	Mat RGB_to_HSI(Mat&image);//BGRtoHSIɫ��ת��
	Mat HSI_to_RGB(Mat&image);//HSItoBGRɫ��ת��

	void pixel_read(Mat&image);//ͨ��ָ�뷽ʽ�������ص�

	void saving(Mat&image);


	const String origin_image = "ԭͼ��";
	const String result_image = "��ʾ���";


void ColorReduce(Mat& input, Mat& output, int div);
void on_trackbar(int pos, void* userdata);

void test01();//���˿ؼ�
void test02();//RGBתHSI
void test03();//HSIתRGB

void test04();//�Ҷ�ֱ��ͼ��ͳ��
void test06();//ʵ�־�ֵ�˲�����˹ƽ��
void test07();//ͼ����

void test05();//ͼ�񼸺α任����תƽ��

void test08();//ʵ�ָ���Ҷ�任

void test09();//������ֵ��
void test10();//����������
void test11();//Canny�㷨
void test12();//����任(ʶ��ֱ�ߺ�Բ)
void test13();//�ǵ���









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
}*///ͨ��ָ��������ص�

