#include"head.h"

string imagename;

void menu() {
	cout<<"说明：输入序号\n"
		<< "1、滑杆控件\n" 
		<< "2、RGB转HSI\n" 
		<< "3、HSI转RGB\n" 
		<< "4、生成灰度直方图统计文件（test.txt）和均衡化后的图像\n"
		<< "5、图像几何变换—旋转\n"
		<< "6、均值滤波\n" 
		<< "7、图像锐化\n"
		<< "8、离散傅里叶变换（DFT）\n"
		<< "9、迭代阈值法\n"
		<< "10、区域生长\n"
		<< "11、Canny算法\n"
		<< "12、哈夫变换\n"
		<< "13、角点检测\n"
        << "0、退出" << endl;

}
int main() {
	int choice = 0;

	//读取图片，默认为Lena图片
	/*cout << "请输入图片名" << endl;
	cin >> imagename;*/

	menu();
	cin >> choice;
	//choice = 7;
	switch (choice)
	{
	case 1:test01(); break;
	case 2: test02(); break;
	case 3: test03(); break;
	case 4: test04(); break;
	case 5:test05(); break;
	case 6:test06(); break;
	case 7:test07(); break;
	case 8:test08(); break;
	case 9:test09(); break;
	case 10:test10(); break;
	case 11:test11(); break;
	case 12:test12(); break;
	case 13:test13(); break;
	
	break;
	}
		
		waitKey(0);
	
	return 0;
}