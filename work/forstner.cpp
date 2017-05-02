#include <iostream>
#include <opencv2/opencv.hpp>

#include "../include/help_func.h"
#include "../include/subImgIterator.h"
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

	Mat robert_img_utemp;
	Mat robert_img_vtemp;
	// filter2D(img,robert_img_utemp,CV_32F,robert_kenerlu,Point(0,0));
	// filter2D(img,robert_img_vtemp,CV_32F,robert_kenerlv,Point(0,0));
	// Mat robert_img_u=Mat(robert_img_utemp,
	// 					Rect(0,0,img_width-1,img_height-1));
	// Mat robert_img_v=Mat(robert_img_vtemp,
	// 					Rect(0,0,img_width-1,img_height-1));
	robert(img,robert_img_utemp,robert_img_vtemp);
	//===========get the w_mat_all and q_mat_all
	cout<<"get the w_mat_all and q_mat_all:begin"<<endl;
	SubImgIterator u_begin(Point(0,0),&robert_img_utemp,2,true);
	SubImgIterator v_begin(Point(0,0),&robert_img_vtemp,2,true);
	auto u_end=u_begin.get_end();
	auto v_end=v_begin.get_end();


	Mat w_mat_all=Mat::zeros(img_height-1,img_width-1,CV_32FC1);
	Mat q_mat_all=Mat::zeros(img_height-1,img_width-1,CV_32FC1);
	Mat w_mat=Mat(w_mat_all,Rect(2,2,w_mat_all.cols-3,w_mat_all.rows-3));
	Mat q_mat=Mat(q_mat_all,Rect(2,2,q_mat_all.cols-3,q_mat_all.rows-3));

	auto w_p=w_mat.begin<float>();
	auto q_p=q_mat.begin<float>();

	while(u_begin != u_end)
	{
		assert(v_begin != v_end);
		float uu=calculate_uu(*u_begin);
		// if(uu!=0) 
		// 	cout<<"uu isn't 0"<<endl;
		float vv=calculate_vv(*v_begin);
		float uv=calculate_uv(*u_begin,*v_begin);
		float det=abs(uu*vv-uv*uv);
		float trace=uu+vv;
		float w=0;float q=0;
		if(trace!=0)
		{
			w=det/trace;
			q=4*det/(trace*trace);
		}
		*w_p=w;
		*q_p=q;
		++w_p; // wish the cache is continus
		++q_p;
		++u_begin;
		++v_begin;
	}
	cout<<"get the w_mat_all and q_mat_all:begin"<<endl;

	//===========filter the w_mat_all q_mat_all
	cout<<"filter the w_mat_all and q_mat_all:begin"<<endl;
	Mat filter_mask=Mat::zeros(w_mat_all.rows,w_mat_all.cols,CV_8U);
	float sm=sum(w_mat_all)[0];
	float avge=sm/(w_mat_all.rows*w_mat_all.cols);
	auto w_begin=w_mat_all.begin<float>();
	auto w_end=w_mat_all.end<float>();
	auto q_begin=q_mat_all.begin<float>();
	auto q_end=q_mat_all.end<float>();
	auto filter_begin=filter_mask.begin<uchar>();
	auto filter_end=filter_mask.end<uchar>();
	for(;w_begin!=w_end;++w_begin,++q_begin,++filter_begin)
	{
		if(*q_begin>0.75 && *w_begin>5*avge)
		{
			*filter_begin=1;
		}
	}
	Mat w_mat_all_new;
	w_mat_all.copyTo(w_mat_all_new,filter_mask);//need to confirm this function does work
	cout<<"filter the w_mat_all and q_mat_all:end"<<endl;

	//===============use the big windwo(7*7) to locate the points
	SubImgIterator new_w_begin(Point(0,0),&w_mat_all_new,3,false);
	auto new_w_end=new_w_begin.get_end();
	vector<Point> points;
	size_t clock=0;
	for(;new_w_begin!=new_w_end;++new_w_begin)
	{
		auto pos=new_w_begin.get_left_top();
		Point max_point;
		Point min_point;
		double max_val;
		double min_val;
		minMaxLoc(*new_w_begin,&min_val,&max_val,&min_point,&max_point);
		++clock;
		if(max_val!=min_val)
			points.push_back(max_point+pos); //we can store the max_val using std::pair,I don't want to do this
	}
	for(auto & point : points)
	{
		circle(img,point,5,Scalar(255,0,0));
	}
	namedWindow("img",WINDOW_NORMAL);
	imshow("img",img);
	imwrite("forstner_dst.jpg",img);
	waitKey();
	// waitKey(0);
	// Point p1(1,2);
	// Point p2(3,4);
	// cout<<"Point:"<<p1+p2<<endl;
	return 0;
}



