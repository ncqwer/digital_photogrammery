#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

using namespace cv;
using namespace std;


double correlation(const cv::Mat& roi_l,
				   const cv::Mat& roi_r);

double matchOnCorrelation(const cv::Mat& left,
						  const cv::Mat& right,
						  const cv::Point& ptl,
						  cv::Point& ptr,
						  size_t half_window);

void getPointPos(const vector<Point2f>& ptls,
				 const Eigen::Vector3f& para_as,
				 vector<Point2f>& ptrs);

void getROI_r(const cv::Mat& right,
			  const vector<cv::Point2f>& ptrs,
			  cv::Mat& roi_r);

void calculate_A(const cv::Mat& right,
				 const cv::Point2f& center,
				 const cv::Point2f& ptl,
				 const cv::Point2f& ptr,
				 Eigen::Matrix<double,1,5>& A);

void calculate_L(float v_l,float v_r,
				 const Eigen::Vector2f& para_hs,
				 double& L);
void update(const Eigen::Matrix<double,5,1>& delta,
			Eigen::Vector2f& para_hs,
			Eigen::Vector3f& para_as);

void least_square_optimize(const cv::Mat& left,
						   const cv::Mat& right,
						   const cv::Point& ptl,
						   const cv::Point& ptr,
						   const int half_window,
						   cv::Point2f& precise_ptl,
						   cv::Point2f& precise_ptr);

int main(int argc, char const *argv[])
{
	int half_window=3;
	// Mat imgl=imread(argv[1],CV_LOAD_IMAGE_GRAYSCALE);
	Mat imgl=imread("l.bmp",CV_LOAD_IMAGE_GRAYSCALE);
	// Mat imgr=imread(argv[2],CV_LOAD_IMAGE_GRAYSCALE);
	Mat imgr=imread("r.bmp",CV_LOAD_IMAGE_GRAYSCALE);

	//get harris corners
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;
	Mat corners_mat;
	cornerHarris( imgl, corners_mat, blockSize, apertureSize, k, BORDER_DEFAULT );

	/// Normalizing
	normalize( corners_mat, corners_mat, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
	convertScaleAbs(corners_mat, corners_mat);

	/// get corners
	vector<Point> ptls;
	double threshould=150;
	int cn = corners_mat.channels();
	for(int j = half_window; j < corners_mat.rows-half_window; j++ )
    {
    	uchar* p = corners_mat.ptr<uchar>(j); 
    	for(int i = half_window; i < corners_mat.cols-half_window; i++ )
        {
            if(*(p+i*cn)>threshould)
            {
            	ptls.push_back(Point(i,j));
            }
        }
    }
	
	//get point on right
	vector<Point> ptrs;
	vector<DMatch> matches;
	size_t count = 0;
	for(auto &ptl : ptls)
	{
		Point ptr;
		std::cout<<"============================="<<std::endl;
		double distance=matchOnCorrelation(imgl,imgr,ptl,ptr,half_window);
		cv::Point2f precise_ptl;
		cv::Point2f precise_ptr;
		if(distance > 0.90)
			least_square_optimize(imgl,imgr,ptl,ptr,half_window,precise_ptl,precise_ptr);
		ptrs.push_back(ptr);
		matches.push_back(DMatch(count,count,distance));
		++count;
	}
	Mat out;
	vector<DMatch> good_matches;
	//get threshould
	std::vector<double> values;
	double values_sum=0;
	for(auto & match : matches)
	{
		values_sum+=match.distance;
		values.push_back(match.distance);
	}
	auto value_size = values.size();
	auto average_thresh = values_sum/value_size;
	auto middle_thresh = average_thresh;
	for(auto & match : matches)
	{
		if(abs(match.distance)>0.95)
		{
			good_matches.push_back(match);
		}
	}
	//transform pts to kpts
	vector<KeyPoint> kptls;
	vector<KeyPoint> kptrs;
	for(auto &pt : ptls)
	{
		kptls.push_back(KeyPoint(pt.x,pt.y,5));
	}
	for(auto &pt : ptrs)
	{
		kptrs.push_back(KeyPoint(pt.x,pt.y,5));
	}
	drawMatches(imgl,kptls,imgr,kptrs,good_matches,out);
	namedWindow("out",WINDOW_NORMAL);
	imshow("out",out);
	waitKey();
	return 0;
}


double correlation(const cv::Mat& roi_l,
				   const cv::Mat& roi_r)
{
	int width=roi_l.cols;
	int height=roi_l.rows;
	Mat roi_lf;
	Mat roi_rf;
	roi_l.convertTo(roi_lf,CV_32FC1);
	roi_r.convertTo(roi_rf,CV_32FC1);
	// //multiply_sum
	// Mat mat_multiply;
	// multiply(roi_l,roi_r,mat_multiply);
	// double multiply_sum = sum(mat_multiply)[0];

	// //l_sum && r_sum
	// double l_sum = sum(roi_l)[0];
	// double r_sum = sum(roi_r)[0];

	// //square_sum
	// Mat square_l;
	// multiply(roi_l,roi_l,square_l);
	// double square_sum_l = sum(square_l)[0];
	// Mat square_r;
	// multiply(roi_r,roi_r,square_r);
	// double square_sum_r = sum(square_r)[0];

	double area = width*height;
	// double distance = (multiply_sum - l_sum*r_sum/area)/(sqrt((square_sum_l - l_sum*l_sum/area)*(square_sum_r - r_sum*r_sum/area))); 
 	auto sum_l = sum(roi_lf)/area;
 	auto sum_r = sum(roi_rf)/area;
 	Mat temp_l = roi_lf - sum_l;
 	Mat temp_r = roi_rf - sum_r;
 	Mat mul;
 	Mat square_l;
 	Mat square_r;
 	multiply(temp_l,temp_r,mul);
 	multiply(temp_l,temp_l,square_l);
 	multiply(temp_r,temp_r,square_r);
 	double distance = (sum(mul)[0])/(sqrt((sum(square_l)[0])*(sum(square_r)[0])));
 	return distance;	
}

double matchOnCorrelation(const cv::Mat& left,
						  const cv::Mat& right,
						  const cv::Point& ptl,
						  cv::Point& ptr,
						  size_t half_window = 2)
{
	//find roi in left
	int i = ptl.x;
	int j = ptl.y;
	int window_size=2*half_window+1;
	Mat roi_l(left,Rect(i-half_window,j-half_window,window_size,window_size));
	double max_distance=-100;
	size_t max_x=0;
	for(size_t x = half_window;x < right.cols - half_window;++x)
	{
		//get roi in right
		Mat roi_r(right,Rect(x-half_window,j-half_window,window_size,window_size));
		//calculate 
		double distance=correlation(roi_l,roi_r);
		if(distance>max_distance)
		{
			//record it
			max_distance=distance;
			max_x=x;
		}
	}
	ptr.x=max_x;
	ptr.y=j;
	cout<<"Point_left:"<<ptl<<endl
		<<"Point_right:"<<ptr<<endl
		<<"distance:"<<max_distance<<endl;
	return max_distance;
}

//least square method
//in: left
//in: right
//in: ptl
//in: ptr
//out: precise_ptl
//out: precise_ptr
void least_square_optimize(const cv::Mat& left,
						   const cv::Mat& right,
						   const cv::Point& ptl,
						   const cv::Point& ptr,
						   const int half_window,
						   cv::Point2f& precise_ptl,
						   cv::Point2f& precise_ptr)
{
	//get center
	cv::Point2f center;
	center.x = ptl.x;
	center.y = ptl.y;
	//get defualt a for first iteration
	float a0 = ptr.x-ptl.x;
	float h0 = right.at<uchar>(ptl.y,ptl.x) - left.at<uchar>(ptr.y,ptr.x);
	Eigen::Vector2f para_hs;
	para_hs<<h0,1;
	// std::cout<<"initiall_para_hs:"<<para_hs<<std::endl;
	Eigen::Vector3f para_as;
	para_as<<a0,1,0;
	// std::cout<<"initiall_para_as:"<<para_as<<std::endl;
	double last_distance = -100;
	//get ptls
	vector<Point2f> ptls;
	for(int y = ptl.y - half_window;y < ptl.y + half_window + 1; ++y)
	{
		for(int x = ptl.x - half_window;x < ptl.x + half_window + 1; ++x)
		{
			ptls.push_back(cv::Point2f(x,y));
		}
	}

	int window_size = 2*half_window + 1;
	Mat roi_r;
	roi_r.create(window_size,window_size,CV_8UC1);
	Mat roi_l(left,Rect(ptl.x-half_window,ptl.y-half_window,window_size,window_size));
	bool first = true;
	size_t iter_num = 0;
	while(true)
	{
		// std::cout<<"iter index:"<<iter_num<<std::endl;
		//construct ptrs with right and roi_l's pos
		//in: para_as;
		//in: ptls
		//out: ptrs
		vector<Point2f> ptrs;
		getPointPos(ptls,para_as,ptrs);

		//construct roi_r with ptrs
		//in:ptrs
		//in:right
		//out:roi_r
		getROI_r(right,ptrs,roi_r);
		//convert roi_l and roi_r to float
		Mat roi_lf;
		Mat roi_rf;
		roi_l.convertTo(roi_lf,CV_32FC1);
		roi_r.convertTo(roi_rf,CV_32FC1);

		double distance = correlation(roi_l,roi_r);
		std::cout<<"----------------------"<<std::endl;
		std::cout<<"hello iter_times:"<<iter_num++<<" iteration distance:"<<distance<<std::endl;
		if(distance < last_distance)
		{
			break; //break out iteration
		}
		last_distance = distance;
		//begin least square iteration
		//note: opencv didn't work well with matrix  so use Eigen instead
		Eigen::Matrix<double,5,5> ATA = Eigen::Matrix<double,5,5>::Zero();
		Eigen::Matrix<double,5,1> ATL = Eigen::Matrix<double,5,1>::Zero();
		auto beginl = roi_lf.begin<float>();
		auto endl = roi_lf.end<float>();
		auto beginr = roi_rf.begin<float>();
		auto ptl_begin = ptls.begin();
		auto ptr_begin = ptrs.begin();
		for(;beginl != endl; ++beginl, ++beginr,++ptl_begin,++ptr_begin)
		{
			//get A(size:1*8)
			//in:dx(right) dy(right) 
			//in:ptr in right
			//in:ptl in left
			Eigen::Matrix<double,1,5> A;
			calculate_A(right,center,*ptl_begin,*ptr_begin,A);
			//get L(size:1*1)
			double L=0.0;
			calculate_L(*beginl,*beginr,para_hs,L);

			ATA+=(A.transpose() * A);
			ATL+=(A.transpose()*L);
		}
		// std::cout<<"ATA:"<<std::endl<<ATA<<std::endl;
		// std::cout<<"ATL:"<<std::endl<<ATL<<std::endl;
		Eigen::Matrix<double,5,1> delta = ATA.inverse() * ATL;
		//update para_as,para_hs
		// update(delta,para_hs,para_as);
		para_hs(0) += delta(0);
		para_hs(1) += delta(1);
		para_as(0) += delta(2);
		para_as(1) += delta(3);
		para_as(2) += delta(4);
		std::cout<<"para_as"<<std::endl<<para_as<<std::endl;
		std::cout<<"para_hs"<<std::endl<<para_hs<<std::endl;
	}

	//get precise left point
	float precise_l_x = 0;
	float precise_l_y = 0;
	float grad_sum = 0;
	for(auto &ptl : ptls)
	{
		float x = ptl.x;
		float y = ptl.y;
		float grad = (right.at<uchar>(y,x+1)-right.at<uchar>(y,x-1))/2;
		precise_l_x +=x*grad;
		precise_l_y +=y*grad;
		grad_sum+=grad;
	}
	precise_l_x/=grad_sum;
	precise_l_y/=grad_sum;
	//get precise right point
	float precise_r_x = para_as(0) + para_as(1)*precise_l_x + para_as(2)*precise_l_y;
	float precise_r_y = precise_l_y;

	precise_ptl.x=precise_l_x;
	precise_ptl.y=precise_l_y;
	precise_ptr.x=precise_r_x;
	precise_ptr.y=precise_r_y;
}

void getPointPos(const vector<Point2f>& ptls,
				 const Eigen::Vector3f& para_as,
				 vector<Point2f>& ptrs)
{
	ptrs.clear();

	for(auto &ptl : ptls)
	{
		float x_r = para_as(0) + para_as(1)*ptl.x + para_as(2)*ptl.y;
		if(x_r<0 && x_r >= 1240)
			std::cout<< x_r<<std::endl;
		float y_r = ptl.y;
		ptrs.push_back(cv::Point2f(x_r,y_r)); 
	}
}

void getROI_r(const cv::Mat& right,
			  const vector<cv::Point2f>& ptrs,
			  cv::Mat& roi_r)
{
	auto begin = roi_r.begin<uchar>();
	for(auto &ptr : ptrs)
	{
		double y = ptr.y;
		double x = ptr.x;
		if(ptr.x >= right.cols)
		{
			x = double(right.cols);
		}
		double x_left = int(x);
		double x_right = int(x) + 1;
		double value_left = right.at<uchar>(y,x_left);
		double value_right = right.at<uchar>(y,x_right);
		double value = (1 - x + x_left)*value_left + (1 + x - x_right)*value_right;
		*begin = value;
		++begin;
	}
}

void calculate_A(const cv::Mat& right,
				 const cv::Point2f& center,
				 const cv::Point2f& ptl,
				 const cv::Point2f& ptr,
				 Eigen::Matrix<double,1,5>& A)
{
	int x_l = ptl.x /*- center.x*/;
	int y_l = ptl.y /*- center.y*/;
	int x_r = ptr.x;
	int y_r = ptr.y;
	float dx_r = (right.at<uchar>(y_r,x_r+1)-right.at<uchar>(y_r,x_r-1))/2;
	// float dy_r = right.at<uchar>(y_r+1,x_r)-right.at<uchar>(y_r,x_r)
	float c1 = 1;
	float c2 = right.at<float>(y_r,x_r); // need to discussion
	float c3 = dx_r;
	float c4 = x_l*dx_r;
	float c5 = y_l*dx_r;
	A<<c1,c2,c3,c4,c5;
}

void calculate_L(float v_l,float v_r,
				 const Eigen::Vector2f& para_hs,
				 double& L)
{
	double v = v_l - (para_hs(0)+para_hs(1)*v_r);
	L = v;
}
void update(const Eigen::Matrix<double,5,1>& delta,
			Eigen::Vector2f& para_hs,
			Eigen::Vector3f& para_as)
{
	float h0 = delta(0) + (delta(1) + 1)*para_hs(0);
	float h1 = (delta(1) + 1)*para_hs(1);
	para_hs(0) = h0;
	para_hs(1) = h1;

	float a0 = delta(2) + (delta(3) + 1)*para_as(0);
	float a1 = (delta(3) + 1)*para_as(1);
	float a2 = (delta(3) + 1)*para_as(2) + delta(4);
	para_as(0) = a0;
	para_as(1) = a1;
	para_as(2) = a2;
}

double VectorScalarProduct(const cv::Mat& roi_l,
						   const cv::Mat& roi_r)
{
	cv::Mat roi_lf,roi_rf,temp;
	roi_l.convertTo(roi_lf,CV_32FC1);
	roi_r.convertTo(roi_rf,CV_32FC1);
	cv::multiply(roi_l,roi_r,temp);
	return cv::sum(temp)[0];
}

double VectorScalarProject(const cv::Mat& roi_l,
						   const cv::Mat& roi_r)
{
	cv::Mat roi_lf,roi_rf,temp;
	roi_l.convertTo(roi_lf,CV_32FC1);
	roi_r.convertTo(roi_rf,CV_32FC1);

	auto width = roi_l.cols;
	auto height = roi_l.rows;
	auto area = width*height;

	auto e_l = cv::sum(roi_lf)[0]/area;
	auto e_r = cv::sum(roi_rf)[0]/area;

	cv::Mat temp_l = roi_lf - e_l;
	cv::Mat temp_r = roi_rf - e_r;
	cv::multiply(temp_l,temp_r,temp);
	return cv::sum(temp)[0];
}

double DiffSquareSum(const cv::Mat& roi_l,
					 const cv::Mat& roi_r)
{
	cv::Mat roi_lf,roi_rf,temp;
	roi_l.convertTo(roi_lf,CV_32FC1);
	roi_r.convertTo(roi_rf,CV_32FC1);

	cv::Mat diff = roi_lf - roi_rf;
	cv::multiply(diff,diff,temp);
	return cv::sum(temp)[0];
}

double DiffAbsoluteSum(const cv::Mat& roi_l,
					   const cv::Mat& roi_r)
{
	cv::Mat roi_lf,roi_rf,temp;
	roi_l.convertTo(roi_lf,CV_32FC1);
	roi_r.convertTo(roi_rf,CV_32FC1);

	cv::absdiff(roi_lf,roi_rf,temp);
	return cv::sum(temp)[0];
}