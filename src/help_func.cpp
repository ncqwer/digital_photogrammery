#include "../include/help_func.h"

bool operator==(const SubImgIterator& lhs, const SubImgIterator& rhs)
{
	return (lhs._left_top == rhs._left_top) && (lhs._img==rhs._img);
}


bool operator!=(const SubImgIterator& lhs, const SubImgIterator& rhs)
{
	return !(lhs==rhs);
}


float calculate_uu(const cv::Mat& mat)
{
	return calculate_uv(mat,mat);
}

float calculate_vv(const cv::Mat& mat)
{
	return calculate_uv(mat,mat);
}

float calculate_uv(const cv::Mat& lhs,const cv::Mat& rhs)
{
	cv::Mat dst;
	cv::Mat lhs_roi=cv::Mat(lhs,cv::Rect(0,0,lhs.cols-1,lhs.rows-1));
	cv::Mat rhs_roi=cv::Mat(lhs,cv::Rect(0,0,rhs.cols-1,rhs.rows-1));
	cv::multiply(lhs_roi,rhs_roi,dst);
	auto ans1=cv::sum(dst);
	auto ans2=ans1[0];
	return ans2;
	// return cv::sum(dst)[0];
}

void robert(const cv::Mat& img, cv::Mat& u_mat, cv::Mat& v_mat)
{
	// size_t img_width=img.cols;
	// size_t img_height=img.rows;
	cv::Mat robert_kenerlu=(cv::Mat_<float>(2,2)<<-1,0,0,1);
	cv::Mat robert_kenerlv=(cv::Mat_<float>(2,2)<<0,-1,1,0);

	filter2D(img,u_mat,CV_32F,robert_kenerlu,cv::Point(0,0));
	filter2D(img,v_mat,CV_32F,robert_kenerlv,cv::Point(0,0));
}

void xy_diff(const cv::Mat& img, cv::Mat& x_mat, cv::Mat& y_mat)
{
	cv::Mat diff_kenerlx=(cv::Mat_<float>(1,2)<<1,-1);
	cv::Mat diff_kenerly=(cv::Mat_<float>(2,1)<<1,-1);

	filter2D(img,x_mat,CV_32F,diff_kenerlx,cv::Point(0,0));
	filter2D(img,y_mat,CV_32F,diff_kenerly,cv::Point(0,0));

}

