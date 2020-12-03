#include"head.h"

//读取照片
Mat my_image = imread("lena.bmp",1);
Mat img = imread("lena.bmp", IMREAD_GRAYSCALE);//用于测试八
//Mat my_image = imread(imagename);//imagename：输入图片名

//增亮函数
void  operators_demo(Mat& image) {
	Mat dst;
	dst = image + Scalar(20, 20, 20);
	imshow("原图", image);
	imshow("加", dst);
}

Mat test07cal(Mat& d, int k[9], int sum) {

	Mat result(my_image.rows, my_image.cols, CV_8UC3);

	for (int i = 0; i < d.rows; i++) {
		for (int j = 0; j < d.cols; j++) {
			if ((i - 1) >= 0 && (j - 1) >= 0 && (i + 1) < d.rows && (j + 1) < d.cols) {
				//BOX模板
				result.at<Vec3b>(i, j)[0] = (k[4] * d.at<Vec3b>(i, j)[0] +
					k[0] * d.at<Vec3b>(i - 1, j - 1)[0] +
					k[1] * d.at<Vec3b>(i - 1, j)[0] +
					k[2] * d.at<Vec3b>(i - 1, j + 1)[0] +
					k[3] * d.at<Vec3b>(i, j - 1)[0] +
					k[5] * d.at<Vec3b>(i, j + 1)[0] +
					k[6] * d.at<Vec3b>(i + 1, j - 1)[0] +
					k[7] * d.at<Vec3b>(i + 1, j)[0] +
					k[8] * d.at<Vec3b>(i + 1, j + 1)[0]) / sum;
				result.at<Vec3b>(i, j)[1] = (k[4] * d.at<Vec3b>(i, j)[1] +
					k[0] * d.at<Vec3b>(i - 1, j - 1)[1] +
					k[1] * d.at<Vec3b>(i - 1, j)[1] +
					k[2] * d.at<Vec3b>(i - 1, j + 1)[1] +
					k[3] * d.at<Vec3b>(i, j - 1)[1] +
					k[5] * d.at<Vec3b>(i, j + 1)[1] +
					k[6] * d.at<Vec3b>(i + 1, j - 1)[1] +
					k[7] * d.at<Vec3b>(i + 1, j)[1] +
					k[8] * d.at<Vec3b>(i + 1, j + 1)[1]) / sum;
				result.at<Vec3b>(i, j)[2] = (k[4] * d.at<Vec3b>(i, j)[2] +
					k[0] * d.at<Vec3b>(i - 1, j - 1)[2] +
					k[1] * d.at<Vec3b>(i - 1, j)[2] +
					k[2] * d.at<Vec3b>(i - 1, j + 1)[2] +
					k[3] * d.at<Vec3b>(i, j - 1)[2] +
					k[5] * d.at<Vec3b>(i, j + 1)[2] +
					k[6] * d.at<Vec3b>(i + 1, j - 1)[2] +
					k[7] * d.at<Vec3b>(i + 1, j)[2] +
					k[8] * d.at<Vec3b>(i + 1, j + 1)[2]) / sum;
			}
			else {
				result.at<Vec3b>(i, j)[0] = d.at<Vec3b>(i, j)[0];
				result.at<Vec3b>(i, j)[1] = d.at<Vec3b>(i, j)[1];
				result.at<Vec3b>(i, j)[2] = d.at<Vec3b>(i, j)[2];
			}
		}
	}
	return result;
}


//HSI/RGB转换函数
Mat RGB_to_HSI(Mat&image) {

	int row = image.rows;
	int col = image.cols, i, j;
	Mat rmat(row, col, CV_64FC3);
	int count = 0;

	double b, g, r, a1, b1, th;

	for (i = 0; i < row; i++) {
		for (j = 0; j < col; j++) {

            double minimum, h, s, i1;
			Vec3b bgr = image.at<Vec3b>(i, j);
           
			b = bgr[0];
			g = bgr[1];
			r = bgr[2];

			a1 = (r - g + r - b) * 0.5;
			b1 = sqrt((r - g) * (r - g) + (r - b) * (g - b));
			if (b1 == 0)h = 0;
			th = a1 / b1;
			th = acos(th);

			if (g >= b) {
				h = th;
			}
			else if (g < b) {
				h = 2 * CV_PI - th;
			}

			minimum = min(r, min(g, b));

			h = h / (2 * CV_PI);
			s = 1 - 3.0 * minimum/(r+g+b);
			i1 = (r + g + b) / (3.0*255.0);
			
			rmat.at<Vec3d>(i, j)[0] = h;  //B
			rmat.at<Vec3d>(i, j)[1] = s;  //G
			rmat.at<Vec3d>(i, j)[2] = i1;   //R
		}
	}
	imshow(origin_image, image);
	return rmat;
}
Mat HSI_to_RGB(Mat&image) {

		int row = image.rows;
		int col = image.cols;
		Mat dst(row, col, CV_64FC3);

		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				double preh = image.at<Vec3b>(i, j)[0] * 2.0 * CV_PI;//H
				double pres = image.at<Vec3b>(i, j)[1];  //S
				double prei = image.at<Vec3b>(i, j)[2];  //I
				double r = 0, g = 0, b = 0;
				double t1, t2, t3;
				t1 = (1.0 - pres) / 3.0;
				if (preh >= 0 && preh < (CV_PI * 2 / 3)) {
					b = t1;
					t2 = pres * cos(preh);
					t3 = cos(CV_PI / 3 - preh);
					r = (1 + t2 / t3) / 3;
					r = 3 * prei * r;
					b = 3 * prei * b;
					g = 3 * prei - (r + b);
				}
				else if (preh >= (CV_PI * 2 / 3) && preh < (CV_PI * 4 / 3)) {
					r = t1;
					t2 = pres * cos(preh - 2 * CV_PI / 3);
					t3 = cos(CV_PI - preh);
					g = (1 + t2 / t3) / 3;
					r = 3 * prei * r;
					g = 3 * g * prei;
					b = 3 * prei - (r + g);
				}
				else if (preh >= (CV_PI * 4 / 3) && preh <= (CV_PI * 2)) {
					g = t1;
					t2 = pres * cos(preh - 4 * CV_PI / 3);
					t3 = cos(CV_PI * 5 / 3 - preh);
					b = (1 + t2 / t3) / 3;
					g = 3 * g * prei;
					b = 3 * prei * b;
					r = 3 * prei - (g + b);
				}
				dst.at<Vec3d>(i, j)[0] = b;
				dst.at<Vec3d>(i, j)[1] = g;
				dst.at<Vec3d>(i, j)[2] = r;
			}
		}
		imshow(origin_image, image);
		imshow(result_image, dst);
		return dst;
	}

//像素读取函数 这个函数可以以指针的方式按行读取，速度更快
void  pixel_read(Mat&image) {
	int w = image.cols;
	int h = image.rows;
	int dim = image.channels();

	for (int row = 0; row < h; row++) {
		uchar* current_row = image.ptr<uchar>(row);
		for (int col = 0; col < w; col++) {
			if (dim == 3) {
				//此处放置对RGB灰度处理的代码
			}
		}
	}
	imshow("output", image);
}


//保存函数
void saving(Mat&im) {
	imwrite("输出图像", im);
}

/////////////////////////滑杆控件/////////////////////////////////////
//滑杆控件实现函数
void ColorReduce(Mat& input, Mat& output, int div) {
	output = input.clone();
	int row = output.rows;
	int col = output.cols;
	int cha = output.channels();

	if (output.isContinuous()) {
		col = col * row * cha;
		row = 1;
	}

	for (int i = 0; i < row; i++) {
		uchar* data = output.ptr<uchar>(i);
		for (int j = 0; j < col; j++) {
			data[j] = data[j] / div * div + div / 2;
		}
	}
}
//回调函数
void on_trackbar(int pos, void* userdata) {
	Mat result;
	if (pos <= 0)result = my_image;
	else ColorReduce(my_image, result, pos);
	imshow("result", result);
}



//滑杆控件
void test01() {
	namedWindow("ori"); namedWindow("result");
	int slider = 0;
	createTrackbar("ColorReduce", "result", &slider, slidermax, on_trackbar);
	imshow("ori", my_image);
	imshow("result", my_image);
}
////////////////////////////滑杆控件//////////////////////////////////

//RGB转HSI
void test02() {
	
	imshow("test2", RGB_to_HSI(my_image));
}

//HSI转RGB
void test03() {
	
	Mat t;
	cvtColor(my_image, t, COLOR_HSV2RGB);
	imshow("test3", t);
}

//灰度直方图的统计，直方图的均衡化
void test04() {
	fstream file;
	file.open("text.txt", ios::out);
	if (!file) { cerr << "open wrong!" << endl; }

	int L=256, pHist_B[256], pHist_G[256], pHist_R[256];
	for (int i = 0; i < L; i++) { 
		pHist_B[i] = 0;
		pHist_G[i] = 0;
		pHist_R[i] = 0;
	}
	
	for (int i = 0; i < my_image.rows; i++) {
		for (int j = 0; j < my_image.cols; j++) {
			pHist_B[my_image.at<Vec3b>(i, j)[0]]++;
			pHist_G[my_image.at<Vec3b>(i, j)[1]]++;
			pHist_R[my_image.at<Vec3b>(i, j)[2]]++;
		}
	}
	
	for (int i = 0; i < L; i++) {
		file<<pHist_B[i]<<";"<<pHist_G[i]<<";"<<pHist_R[i]<<endl;
	}

	file.close();

	Mat result(my_image.rows, my_image.cols, CV_8UC3);

	int map[3][256]; double prob[256]; int sum = 0; double acc = 0;
	//B通道均衡化
	for (int i = 0; i < L; i++)sum = sum + pHist_B[i];
	for (int i = 0; i < L; i++)prob[i] = double(pHist_B[i]) / double(sum);
	for (int i = 0; i < L; i++) {
		acc = acc + prob[i];
		map[0][i] = int((L - 1) * acc + 0.5);
	}
	for (int i = 0; i < my_image.rows; i++) {
		for (int j = 0; j < my_image.cols; j++) {
			int bchannel = int(my_image.at<Vec3b>(i, j)[0]);
			result.at<Vec3b>(i, j)[0] = map[0][bchannel];
		}
	}
	//G通道均衡化
	sum = 0; acc = 0;
	for (int i = 0; i < L; i++)sum = sum + pHist_G[i];
	for (int i = 0; i < L; i++)prob[i] = (double)pHist_G[i] / (double)sum;
	for (int i = 0; i < L; i++) {
		acc = acc + prob[i];
		map[1][i] = int((L - 1) * acc + 0.5);
	}
	for (int i = 0; i < my_image.rows; i++) {
		for (int j = 0; j < my_image.cols; j++) {
			int bchannel = int(my_image.at<Vec3b>(i, j)[1]);
			result.at<Vec3b>(i, j)[1] = map[1][bchannel];
		}
	}
	//R通道均衡化
	sum = 0; acc = 0;
	for (int i = 0; i < L; i++)sum = sum + pHist_R[i];
	for (int i = 0; i < L; i++)prob[i] = (double)pHist_R[i] /(double) sum;
	for (int i = 0; i < L; i++) {
		acc = acc + prob[i];
		map[2][i] = int((L - 1) * acc + 0.5);
	}
	for (int i = 0; i < my_image.rows; i++) {
		for (int j = 0; j < my_image.cols; j++) {
			int bchannel = int(my_image.at<Vec3b>(i, j)[2]);
			result.at<Vec3b>(i, j)[2] = map[2][bchannel];
		}
	}
	
	
	imshow("均衡化结果", result);

	//opencv自带函数均衡化结果
	vector<Mat> re;
	split(my_image, re);
	equalizeHist(re[0], re[0]);
	equalizeHist(re[1], re[1]);
	equalizeHist(re[2], re[2]);
	merge(re, result);
	imshow("函数结果", result);
	imwrite("result_1.jpg", result);
}

//图像的几何变换—旋转（逆时针旋转）
void test05() {
	int angle = 45; double degree = CV_PI * angle / 180;
	double row = my_image.rows , col = my_image.cols ;
	Point2f centre(col/2, row/2);
	Mat rotmat = getRotationMatrix2D(centre, angle, 1);
	
	//自定义变换矩阵t实现了图形的旋转与平移，t:2*3矩阵
	Mat t= Mat::zeros(2, 3,CV_32FC1 );//CV_32FC1
		t.at<float>(0, 0) = cos(degree);
		t.at<float>(0, 1) = sin(degree);
		t.at<float>(0, 2) =0;//水平偏移量
		t.at<float>(1, 0) = (-1) *sin(degree);
		t.at<float>(1, 1) = cos(degree);
		t.at<float>(1, 2) = col*sin(degree);//竖直偏移量col*sin(degree)

	int newrow, newcol;
	newrow = int(row * cos(degree) + col * sin(degree));
	newcol = int(row * sin(degree) + col * cos(degree));
	Mat result1(newrow, newcol, my_image.type());
	Mat result2(row, col, my_image.type());

	//仿射变换
	//自定义矩阵变换
	warpAffine(my_image, result1, t, result1.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
	namedWindow("旋转后图像(未经裁减)", WINDOW_AUTOSIZE);
	imshow("旋转后图像(未经裁减)", result1);
	//getRotationMatrix2D函数
	warpAffine(my_image, result2, rotmat, my_image.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
	namedWindow("旋转后图像", WINDOW_AUTOSIZE);
	imshow("旋转后图像", result2);
}

////实现均值滤波、高斯平滑
//卷积运算函数   input，模板(卷积核)，模板系数，通道数
Mat testcal(Mat& d, int k[9], int sum,int channel) {
	Mat re = d.clone();
	if (channel == 3) {
		Mat result(my_image.rows, my_image.cols, CV_8UC3);

		for (int i = 0; i < d.rows; i++) {
			for (int j = 0; j < d.cols; j++) {
				if ((i - 1) >= 0 && (j - 1) >= 0 && (i + 1) < d.rows && (j + 1) < d.cols) {
					uchar a;
					a = (k[4] * d.at<Vec3b>(i, j)[0] +
						k[0] * d.at<Vec3b>(i - 1, j - 1)[0] +
						k[1] * d.at<Vec3b>(i - 1, j)[0] +
						k[2] * d.at<Vec3b>(i - 1, j + 1)[0] +
						k[3] * d.at<Vec3b>(i, j - 1)[0] +
						k[5] * d.at<Vec3b>(i, j + 1)[0] +
						k[6] * d.at<Vec3b>(i + 1, j - 1)[0] +
						k[7] * d.at<Vec3b>(i + 1, j)[0] +
						k[8] * d.at<Vec3b>(i + 1, j + 1)[0]) / sum;
					if (a < 0)a = 0; else if (a > 255)a = 255;
					result.at<Vec3b>(i, j)[0] = a;
					a = (k[4] * d.at<Vec3b>(i, j)[1] +
						k[0] * d.at<Vec3b>(i - 1, j - 1)[1] +
						k[1] * d.at<Vec3b>(i - 1, j)[1] +
						k[2] * d.at<Vec3b>(i - 1, j + 1)[1] +
						k[3] * d.at<Vec3b>(i, j - 1)[1] +
						k[5] * d.at<Vec3b>(i, j + 1)[1] +
						k[6] * d.at<Vec3b>(i + 1, j - 1)[1] +
						k[7] * d.at<Vec3b>(i + 1, j)[1] +
						k[8] * d.at<Vec3b>(i + 1, j + 1)[1]) / sum;
					if (a < 0)a = 0; else if (a > 255)a = 255;
					result.at<Vec3b>(i, j)[1] = a;
					a = (k[4] * d.at<Vec3b>(i, j)[2] +
						k[0] * d.at<Vec3b>(i - 1, j - 1)[2] +
						k[1] * d.at<Vec3b>(i - 1, j)[2] +
						k[2] * d.at<Vec3b>(i - 1, j + 1)[2] +
						k[3] * d.at<Vec3b>(i, j - 1)[2] +
						k[5] * d.at<Vec3b>(i, j + 1)[2] +
						k[6] * d.at<Vec3b>(i + 1, j - 1)[2] +
						k[7] * d.at<Vec3b>(i + 1, j)[2] +
						k[8] * d.at<Vec3b>(i + 1, j + 1)[2]) / sum;
					if (a < 0)a = 0; else if (a > 255)a = 255;
					result.at<Vec3b>(i, j)[2] = a;
				}
				else {
					result.at<Vec3b>(i, j)[0] = d.at<Vec3b>(i, j)[0];
					result.at<Vec3b>(i, j)[1] = d.at<Vec3b>(i, j)[1];
					result.at<Vec3b>(i, j)[2] = d.at<Vec3b>(i, j)[2];
				}
			}
		}
		return result;
	}
	else if (channel == 1) {
		Mat result(my_image.rows, my_image.cols, CV_8UC1);

		for (int i = 0; i < d.rows; i++) {
			for (int j = 0; j < d.cols; j++) {
				if ((i - 1) >= 0 && (j - 1) >= 0 && (i + 1) < d.rows && (j + 1) < d.cols) {
					uchar a;
					a = (k[4] * d.at<uchar>(i, j) +
						k[0] * d.at<uchar>(i - 1, j - 1) +
						k[1] * d.at<uchar>(i - 1, j) +
						k[2] * d.at<uchar>(i - 1, j + 1) +
						k[3] * d.at<uchar>(i, j - 1) +
						k[5] * d.at<uchar>(i, j + 1) +
						k[6] * d.at<uchar>(i + 1, j - 1) +
						k[7] * d.at<uchar>(i + 1, j) +
						k[8] * d.at<uchar>(i + 1, j + 1)) / sum;
					if (a < 0)a = 0; else if (a > 255)a = 255;
					result.at<uchar>(i, j) = a;
				}
				else {
					result.at<uchar>(i, j) = d.at<uchar>(i, j);
					result.at<uchar>(i, j) = d.at<uchar>(i, j);
					result.at<uchar>(i, j) = d.at<uchar>(i, j);
				}
			}
		}
		return result;
	}
	else return re;
}
//产生椒盐图像
Mat test06salt(Mat&image,int num) {
	Mat re = image.clone();
	int i, j;
	srand(time(NULL));
	for (int k = 0; k < num; k++) {
		i = rand() % image.rows;//产生从0到image.rows的随机数
		j = rand() % image.cols;//产生从0到image.cols的随机数
		re.at<Vec3b>(i, j)[0] = 255;
		re.at<Vec3b>(i, j)[1] = 255;
		re.at<Vec3b>(i, j)[2] = 255;
	}
	return re;
}
//均值滤波、高斯平滑实现函数
void test06() {
	Mat result(my_image.rows, my_image.cols, CV_8UC3);
	Mat d = test06salt(my_image, 2000);

	//均值滤波
		for (int i = 0; i < d.rows; i++) {
			for (int j = 0; j < d.cols; j++) {
				if ((i - 1) >= 0 && (j - 1) >= 0 && (i + 1) < d.rows && (j + 1) < d.cols) {
					//BOX模板
					result.at<Vec3b>(i, j)[0] = (d.at<Vec3b>(i, j )[0] +
						d.at<Vec3b>(i-1, j-1)[0] +
						d.at<Vec3b>(i-1, j )[0] +
						d.at<Vec3b>(i-1, j + 1)[0] +
						d.at<Vec3b>(i , j-1)[0] +
						d.at<Vec3b>(i , j + 1)[0] +
						d.at<Vec3b>(i + 1, j-1)[0] +
						d.at<Vec3b>(i + 1, j  )[0] +
						d.at<Vec3b>(i + 1, j + 1)[0]) / 9;
					result.at<Vec3b>(i, j)[1] = (d.at<Vec3b>(i, j)[1] +
						d.at<Vec3b>(i - 1, j - 1)[1] +
						d.at<Vec3b>(i - 1, j)[1] +
						d.at<Vec3b>(i - 1, j + 1)[1] +
						d.at<Vec3b>(i, j - 1)[1] +
						d.at<Vec3b>(i, j + 1)[1] +
						d.at<Vec3b>(i + 1, j - 1)[1] +
						d.at<Vec3b>(i + 1, j)[1] +
						d.at<Vec3b>(i + 1, j + 1)[1]) / 9;
					result.at<Vec3b>(i, j)[2] = (d.at<Vec3b>(i, j)[2] +
						d.at<Vec3b>(i - 1, j - 1)[2] +
						d.at<Vec3b>(i - 1, j)[2] +
						d.at<Vec3b>(i - 1, j + 1)[2] +
						d.at<Vec3b>(i, j - 1)[2] +
						d.at<Vec3b>(i, j + 1)[2] +
						d.at<Vec3b>(i + 1, j - 1)[2] +
						d.at<Vec3b>(i + 1, j)[2] +
						d.at<Vec3b>(i + 1, j + 1)[2]) / 9;
				}
				else {
					result.at<Vec3b>(i, j)[0] = d.at<Vec3b>(i, j)[0];
					result.at<Vec3b>(i, j)[1] = d.at<Vec3b>(i, j)[1];
					result.at<Vec3b>(i, j)[2] = d.at<Vec3b>(i, j)[2];
				}
			}
		}

		//高斯平滑
		//高斯模板
		int k[9] = { 1,2,1,2,4,2,1,2,1 };
		Mat resultgss=testcal(d, k, 16,3);

	namedWindow("椒盐图像", WINDOW_AUTOSIZE);
	imshow("椒盐图像",d);

	namedWindow("均值滤波", WINDOW_AUTOSIZE);
	imshow("均值滤波", result);

	namedWindow("高斯平滑", WINDOW_AUTOSIZE);
	imshow("高斯平滑", resultgss);

	/*Mat result1(my_image.rows, my_image.cols, my_image.type());
	medianBlur(my_image, result1, 3);
	namedWindow("均值滤波(函数)", WINDOW_AUTOSIZE);
	imshow("均值滤波(函数)", result1);*/
}

//图像锐化
void test07() {

	//读取图片，默认为Lena图片
	/*
	string imagename;
	cout << "请输入图片名" << endl;
	cin >>imagename;
	my_image = imread(imagename, 0);
	*/
	
	/*int k[9] = { -1,-1,-1,-1,5,-1,-1,-1,-1 };*/
	Mat gra;
	//cvtColor(my_image, gra, COLOR_BGR2GRAY);
	Mat result;
	//拉普拉斯算子
	Mat kernel = (Mat_<char>(3, 3) << 0, -1,0, -1, 5, -1,0, -1, 0);
	filter2D(my_image, result, CV_8UC3, kernel);
	imshow("结果", result);

	//Laplacian(gra, resultlap,my_image.depth());
	namedWindow("原图像", WINDOW_AUTOSIZE);
	imshow("原图像", my_image);
	//namedWindow("锐化结果", WINDOW_AUTOSIZE);
	//imshow("锐化结果", resultlap);
}

//实现傅里叶变换
void test08() {
	//在上面定义了
	Mat img = imread("l.jpg", IMREAD_GRAYSCALE);
	Mat padded;
	//扩展图像
	int row = getOptimalDFTSize(img.rows);
	int col = getOptimalDFTSize(img.cols);
	copyMakeBorder(img, padded,0, row - img.rows,0, col - img.cols, BORDER_REPLICATE,Scalar::all(0));
 
    //实部和虚部矩阵的创建
	Mat planes[] = { Mat_<float>(padded),Mat::zeros(padded.size(),CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);
	//实现傅里叶变换
	dft(complexI, complexI);
	//频谱计算
	split(complexI, planes);
	magnitude(planes[0], planes[1], planes[0]);
	Mat magI = planes[0];
	//对数坐标变换，（傅里叶变换结果的动态范围较宽）
	magI += Scalar::all(1);
	log(magI, magI);
	//频谱平移，使频谱的原点位于显示中心
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
	int cx = magI.cols / 2; int cy = magI.rows / 2;
	Mat q0(magI, Rect(0, 0, cx, cy));//左上角
	Mat q1(magI, Rect(cx, 0, cx, cy));//右上角
	Mat q2(magI, Rect(0, cy, cx, cy));//左下角
	Mat q3(magI, Rect(cx, cy, cx, cy));//右下角
	Mat tmp;
	    //实现平移
	q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
	q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);
	//标准化及显示
	normalize(magI, magI, 0, 1, NORM_MINMAX);
	imshow("原图像", img);
	imshow("离散傅里叶变换", magI);
}

///////test09 迭代阈值法
//确定阈值的函数
double iterativeThreshold(const Mat& grayImg)
{
	if (grayImg.empty() || grayImg.depth() != CV_8U)
	{
		cerr << "Invalid image format." << endl;
		return 0;
	}

	/// 计算灰度直方图
	const int histSize[] = { 256 };
	float grayRanges[] = { 0, 256 };
	const float* histRanges[] = { grayRanges };
	int histChannels[] = { 0 };
	Mat_<float> histImg;
	calcHist(&grayImg, 1, histChannels, Mat(), histImg, 1, histSize, histRanges, true, false);

	/// 查找图像中的最小、最大灰度值
	int minGray = 0, maxGray = 255;
	for (minGray = 0; minGray < 256 && histImg(minGray) <= 0; minGray++);
	for (maxGray = 255; maxGray > minGray && histImg(maxGray) <= 0; maxGray--);

	/// 如果图像只有1种或2种灰度，则直接返回最小灰度
	if (maxGray - minGray < 2)
	{
		return minGray;
	}

	/// 以最小与最大灰度值的中间值作为初始阈值
	double thOld = (minGray + maxGray) / 2;

	/// 迭代求解阈值
	double thNew = thOld;
	double leftMean, rightMean, leftSum, rightSum;
	bool bDone = false;
	while (!bDone)
	{
		// 根据上一步的阈值将图像分割为两部分并求出两部分的灰度均值
		leftMean = 0;
		leftSum = 0;
		int i = minGray;
		for (; i <= thOld; i++)
		{
			leftSum += histImg(i);
			leftMean += (double)histImg(i) * (double)i;
		}
		leftMean /= leftSum;

		rightMean = 0;
		rightSum = 0;
		for (; i <= maxGray; i++)
		{
			rightSum += histImg(i);
			rightMean += (double)histImg(i) *(double) i;
		}
		rightMean /= rightSum;

		// 基于两部分的灰度均值更新阈值
		thNew = (leftMean + rightMean) / 2.0;

		bDone = abs(thNew - thOld) < 1.0;
		thOld = thNew;
	}

	return thOld;
}
//迭代阈值法的实现
void test09() {

	Mat result(img.rows, img.cols, img.type());
	double th = iterativeThreshold(img);

	/// 阈值分割
	threshold(img, result, th, 255, THRESH_BINARY);

	imshow("迭代阈值法", result);
}


////////////////////////////////////////////////////////////
//////////////////////实现区域生长//////////////////////////
////////////////////////////////////////////////////////////

const string winName = "RegionGrow"; // 窗口名称
Mat srcImg;	// 原始图像
Mat dstImg;	// 填充图像

/// 区域生长的相关参数
int loDiff = 2;		// 当前像素与邻域像素（生长区域）的灰度差值下限的绝对值
int upDiff = 2;		// 当前像素与邻域像素（生长区域）的灰度差值上限的绝对值
int conType = 4;	// 区域连通性（4或8）
int growMode = 0;	// 区域生长准则（0：根据当前像素与相邻像素之间的灰度差值判断；1：根据当前像素与种子像素之间的灰度差值判断）

/// 鼠标回调函数
static void onMouseEvent(int event, int x, int y, int flags, void* userdata);

/// 打印帮助
static void help();


/**
 * 打印帮助
 */
static void help()
{
	cout << "Hot keys: \n"
		"\tESC - quit the program\n"
		"\tr - restore the original image\n"
		"\tf - switch floodfill mode between floating range and fixed range\n"
		"\t4 - use 4-connectivity mode\n"
		"\t8 - use 8-connectivity mode\n" << endl;
}

/**
 * 鼠标回调函数：先通过鼠标单击获取区域生长的种子点，随后进行区域生长
 * @param event 鼠标事件类型
 * @param x 鼠标事件发生时的水平坐标分量
 * @param y 鼠标事件发生时的垂直坐标分量
 * @param flags 鼠标事件标志
 * @param userdata 附加的用户数据
 */
static void onMouseEvent(int event, int x, int y, int flags, void* userdata)
{
	if (dstImg.empty())
	{
		return;
	}

	/// 单击左键选择区域生长的种子点
	if (event == EVENT_FLAG_LBUTTON)
	{
		// 指定区域生长种子点
		Point seedPoint(x, y);

		// 指定生长区域的颜色
		int r = (unsigned char)theRNG() & 255;
		int g = (unsigned char)theRNG() & 255;
		int b = (unsigned char)theRNG() & 255;
		Scalar regionColor = (dstImg.channels() == 1) ? Scalar(r * 0.299 + g * 0.587 + b * 0.114) : Scalar(r, g, b);

		/// 实现区域生长
		// floodFill函数的填充标志（高8位为填充模式，低8位为连通性类型）
		int fillFlags = conType + 0xFF00 + (growMode == 1 ? FLOODFILL_FIXED_RANGE : 0);
		int regionArea = floodFill(dstImg, seedPoint, regionColor, 0,
			Scalar(loDiff, loDiff, loDiff), Scalar(upDiff, upDiff, upDiff), fillFlags);

		// 显示图像
		imshow(winName, dstImg);
		cout << "Region area: " << regionArea << endl;
	}

	/// 单击右键重置填充图像
	if (event == EVENT_FLAG_RBUTTON)
	{
		srcImg.copyTo(dstImg);

		// 显示图像
		imshow(winName, dstImg);
	}
}
void test10() {
	/// 打印帮助
	help();

	/// 加载原始图像
	string srcFileName;
	cout << "Enter the source file name: ";
	cin >> srcFileName;
	srcImg = imread(srcFileName);
	if (srcImg.empty())
	{
		cerr << "Failed to load the source image." << endl;
		return;
	}
	srcImg.copyTo(dstImg);

	/// 命名图像窗口
	namedWindow(winName);

	/// 创建控制loDiff与upDiff的滑块条
	createTrackbar("loDiff", winName, &loDiff, 128, 0, 0);
	createTrackbar("upDiff", winName, &upDiff, 128, 0, 0);

	/// 设置鼠标回调函数
	setMouseCallback(winName, onMouseEvent, &srcImg);

	/// 显示图像
	bool bLoop = true;
	while (bLoop)
	{
		imshow(winName, dstImg);

		int ch = waitKey(0);
		switch (ch)
		{
		case 27:
			bLoop = false;
			break;
		case '4':
			cout << "4-connectivity mode is set\n";
			conType = 4;
			break;
		case '8':
			cout << "8-connectivity mode is set\n";
			conType = 8;
			break;
		case 'f':
			if (growMode == 0)
			{
				growMode = 1;
				cout << "Fixed range floodfill mode is set\n";
			}
			else
			{
				growMode = 0;
				cout << "Floating range floodfill mode is set\n";
			}
			break;
		default:
			break;
		}
	}
}

///////////////////////⬆⬆⬆⬆⬆⬆⬆⬆⬆///////////////////////////
//////////////////////实现区域生长//////////////////////////
////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////
//////////////////////Canny算法//////////////////////////
/////////////////////////////////////////////////////////
//实现函数
/*
@param src 输入图像
@param dst 输出图像
@param lowThresh 低阈值
@param ratio 高阈值与低阈值之比
*/
void edgeCanny(Mat src, Mat & dst, double lowThresh, double ratio = 3)
	{
		CV_Assert(src.depth() == CV_8U);

		/// 先进行图像平滑
		Mat edge;
		blur(src, dst, Size(3, 3));

		/// 应用Canny算子
		Canny(dst, dst, lowThresh, lowThresh * ratio, 3);
	}
//回调函数
void ontrackbar_11(int pos, void*p) {
	Mat dst;
	edgeCanny(*(Mat*)p, dst, pos, 3);
	imshow("Canny算法边缘检测", dst);
}
void test11() {

	Mat dst;
	string imagename;
	int lt = 0;
	int max = 50;

	cout << "请输入图片名" << endl;
	cin >> imagename;
	my_image = imread(imagename,IMREAD_GRAYSCALE);
	if (my_image.empty())
	{
		cerr << "Failed to load image "  << endl;
		return ;
	}

	Mat* p = &my_image;//定义指向用户数据的指针，回调函数使用。Mat* 指针可以无条件转换成void* 

	namedWindow("Canny算法边缘检测", WINDOW_AUTOSIZE);
	//p:传递给回调函数的数据
	createTrackbar("低阈值", "Canny算法边缘检测", &lt, max, ontrackbar_11,p);
	imshow("Canny算法边缘检测", my_image);
}

/////////////////////////////////////////////////////////
//////////////哈夫变换(直线检测、圆检测)/////////////////
/////////////////////////////////////////////////////////

/*
*Hough变换检测直线
* @param src 原始灰度图像(单通道、8位)
* @param dst 检测结果图像
* @param accumThreshold 累加数组元素阈值（只保留累加数组中大于该阈值的元素）
*/
void stHoughLine(const Mat& src, Mat& dst, int accumThreshold)
{
	CV_Assert(src.depth() == CV_8U);

	/// 初始化输出图像
	cvtColor(src, dst, COLOR_GRAY2BGR);

	/// 按照指定的极径、极角分辨率进行标准Hough变换检测直线，检测结果（rho, theta）保存在lines中
	vector<Vec2f> lines;
	cout << src.channels();
	HoughLines(src, lines, 1, CV_PI / 180, accumThreshold, 0, 0);
	int extLineLen = src.rows + src.cols;
	for(size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0];	// 直线对应的极径
		float theta = lines[i][1];	// 直线对应的极角（极线与x轴的夹角）

		double a = cos(theta);
		double b = sin(theta);

		// 设置当前直线的两个端点坐标（以便显示）
		Point pt1, pt2;
		if(fabs(a) < 0.0001)
		{
			// 当极角接近PI/2时，近似为水平线
			pt1.y = pt2.y = cvRound(rho);
			pt1.x = 0;
			pt2.x = src.cols;
		}
		else if(fabs(b) < 0.0001)
		{
			// 当极角接近0时，近似为垂直线
			pt1.x = pt2.x = cvRound(rho);
			pt1.y = 0;
			pt2.y = src.rows;
		}
		else
		{
			pt1.x = cvRound(rho / a);
			pt1.y = 0;
			pt2.x = cvRound(rho / a - src.rows * b / a);
			pt2.y = src.rows;
		}

		// 绘制当前直线
		line(dst, pt1, pt2, Scalar(0, 0, 255));
	}
}
/*
 * Hough变换检测圆
 * @param src 原始灰度图像(单通道、8位)
 * @param dst 检测结果图像
 * @param accumThreshold 累加数组元素阈值（只保留累加数组中大于该阈值的元素）
 * @param minRadius 需要检测的最小圆半径
 * @param maxRadius 需要检测的最大圆半径
 */
void stHoughCircle(const Mat& src, Mat& dst, int minRadius, int maxRadius)
{
	CV_Assert(src.depth() == CV_8U);
	cvtColor(src, dst, COLOR_GRAY2BGR);
	
	vector<Vec3f> circles;
	HoughCircles(src, circles, HOUGH_GRADIENT, 1, 10, 100, 30, minRadius, maxRadius);

	for (size_t i = 0; i < circles.size(); i++)
	{
		Vec3i c = circles[i];
		circle(dst, Point(c[0], c[1]), c[2], Scalar(0, 0, 255));
	}
}
//直线、圆检测
void test12() {
	Mat dst1,dst2;
	string imagename;
	int order = 0;
	cout << "请输入检测形状：\n1---直线\n2---圆" << endl;
	cin >> order;
	cout << "请输入图片名" << endl;
	cin >> imagename;
	my_image = imread(imagename, IMREAD_GRAYSCALE);
	if (my_image.empty())
	{
		cerr << "Failed to load image " << endl;
		return;
	}

	//进行canny边缘检测
	Canny(my_image, my_image, 50, 100, 3);

	if (order == 1) {
		stHoughLine(my_image, dst1, 100);
		my_image = imread(imagename);
		imshow("原图片", my_image);
		imshow("哈夫变换(直线)", dst1);
	}
	else if (order == 2) {
		stHoughCircle(my_image, dst2, 10, 30);
		my_image = imread(imagename);
		imshow("原图片", my_image);
		imshow("哈夫变换(圆)", dst2);
	}
	else {
		stHoughLine(my_image, dst1, 100);
		my_image = imread(imagename);
		imshow("原图片", my_image);
		imshow("哈夫变换(直线)", dst1);
	}
}

//实现角点检测
void test13() {
	string imagename;
	cout << "请输入图片名" << endl;
	cin >> imagename;
	my_image = imread(imagename);
	if (my_image.empty())
	{
		cerr << "Failed to load image " << endl;
		return;
	}
	if (my_image.channels() >= 3)
		cvtColor(my_image, my_image, COLOR_RGB2GRAY);

	Mat dst,dstNorm,result;
	int blocksize = 3;
	int ksize=3;
	double k = 0.05;
	cornerHarris(my_image, dst, blocksize, ksize, k, BORDER_DEFAULT);
	normalize(dst, dstNorm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dstNorm, result);

	/// Drawing a circle around corners
	my_image = imread(imagename);
	for (int i = 0; i < dstNorm.rows; i++)
	{
		for (int j = 0; j < dstNorm.cols; j++)
		{
			if ((int)dstNorm.at<float>(i, j) > 120)
			{
				circle(my_image, Point(j, i), 2, Scalar(0), 2, 8, 0);
			}
		}
	}

	imshow("result", my_image);
}