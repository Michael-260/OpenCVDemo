#include"head.h"

string imagename;

void menu() {
	cout<<"˵�����������\n"
		<< "1�����˿ؼ�\n" 
		<< "2��RGBתHSI\n" 
		<< "3��HSIתRGB\n" 
		<< "4�����ɻҶ�ֱ��ͼͳ���ļ���test.txt���;��⻯���ͼ��\n"
		<< "5��ͼ�񼸺α任����ת\n"
		<< "6����ֵ�˲�\n" 
		<< "7��ͼ����\n"
		<< "8����ɢ����Ҷ�任��DFT��\n"
		<< "9��������ֵ��\n"
		<< "10����������\n"
		<< "11��Canny�㷨\n"
		<< "12������任\n"
		<< "13���ǵ���\n"
        << "0���˳�" << endl;

}
int main() {
	int choice = 0;

	//��ȡͼƬ��Ĭ��ΪLenaͼƬ
	/*cout << "������ͼƬ��" << endl;
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