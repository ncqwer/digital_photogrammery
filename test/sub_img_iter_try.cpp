#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "../include/subImgIterator.h"
using namespace std;
using namespace cv;

int sub_img_iter_try(int argc, char *argv[])
{
	cv::Mat img=cv::imread(argv[1]);

	SubImgIterator iter(cv::Point(0,0),&img,100,false);

	// namedWindow("1", WINDOW_AUTOSIZE);
	// imshow("1",*iter);
	// namedWindow("2", WINDOW_AUTOSIZE);
	// imshow("2",*(++iter));	
	// namedWindow("3", WINDOW_AUTOSIZE);
	// imshow("3",*(++iter));
	// =====================test end========
	// auto end=iter.get_end();
	// size_t i=0;
	// while(iter!=end)
	// {
	// 	namedWindow(to_string(i));
	// 	imshow(to_string(i),*iter);
	// 	++iter;
	// 	++i;
	// }
	//===================test nest===========
	SubImgIterator iter_nest(Point(0,0),&(*iter),50);
	namedWindow("iter_nest");
	imshow("iter_nest",*iter_nest);
	cout<<"hello world"<<endl;
	waitKey(0);
	return 0;
}