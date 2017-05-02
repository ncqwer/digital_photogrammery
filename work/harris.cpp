#include <iostream>
#include <opencv2/opencv.hpp>

#include "../include/help_func.h"
// #include "../include/subImgIterator.h"
using namespace cv;
using namespace std;

int main(int argc, char const *argv[])
{
	Mat img=imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
	// Mat img=(Mat_<uchar>(3,3)<<1,2,3,4,5,6,7,8,9);
	size_t img_width=img.cols;
	size_t img_height=img.rows;
	// Mat robert_kenerlu=(Mat_<float>(2,2)<<-1,0,0,1);
	// Mat robert_kenerlv=(Mat_<float>(2,2)<<0,-1,1,0);

	Mat x_mat;
	Mat y_mat;
	xy_diff(img,x_mat,y_mat);
	//===========get the w_mat_all and q_mat_all
	Mat x2_mat;
	multiply(x_mat,x_mat,x2_mat);
	Mat y2_mat;
	multiply(y_mat,y_mat,y2_mat);
	Mat xy_mat;
	multiply(x_mat,y_mat,xy_mat);
	GaussianBlur(x2_mat,x2_mat,Size(0,0),0.3);
	GaussianBlur(y2_mat,y2_mat,Size(0,0),0.3);
	GaussianBlur(xy_mat,xy_mat,Size(0,0),0.3);

	Mat w_mat=Mat::zeros(img_height,img_width,CV_32F);
	auto x2_begin=x2_mat.begin<float>();
	auto x2_end=x2_mat.end<float>();
	auto y2_begin=x2_mat.begin<float>();
	auto xy_begin=x2_mat.begin<float>();
	auto w_begin=w_mat.begin<float>();
	for(;x2_begin!=x2_end;++x2_begin,++y2_begin,++xy_begin,++w_begin)
	{
		float det=(*x2_begin)*(*y2_begin)-(*xy_begin)*(*xy_begin);
		float trace=(*x2_begin)*(*x2_begin)+(*y2_begin)*(*y2_begin);
		float I=det-0.04*trace*trace;
		*w_begin=I;
	}
	//===============use the big windwo(7*7) to locate the points
	SubImgIterator new_w_begin(Point(0,0),&w_mat,3,false);
	auto new_w_end=new_w_begin.get_end();
	vector<pair<Point,float>> points;
	for(;new_w_begin!=new_w_end;++new_w_begin)
	{
		auto pos=new_w_begin.get_left_top();
		Point max_point;
		Point min_point;
		double max_val;
		double min_val;
		minMaxLoc(*new_w_begin,&min_val,&max_val,&min_point,&max_point);
		if(max_val!=min_val)
			points.push_back({max_point+pos,max_val}); 
	}
	//==================sort the point
	typedef  pair<Point,float> PF; 
	size_t points_size=points.size();
	size_t num=300<points_size? 300:points_size;
	auto begin=points.begin();
	auto end=points.end();
	auto nth=begin+num;
	nth_element(begin,nth,end,[](const PF&lhs,const PF&rhs){
		return lhs.second<rhs.second;
	});
	for(;nth!=points.end();++nth)
	{
		circle(img,nth->first,5,Scalar(255,0,0));
	}
	namedWindow("img",WINDOW_NORMAL);
	imshow("img",img);
	imwrite("harris_dst.jpg",img);
	waitKey();
	return 0;
}
