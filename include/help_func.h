#ifndef _HELP_FUNC_H_
#define _HELP_FUNC_H_ 

#include <opencv2/opencv.hpp>
#include "subImgIterator.h"

bool operator==(const SubImgIterator& lhs, const SubImgIterator& rhs);
bool operator!=(const SubImgIterator& lhs, const SubImgIterator& rhs);

float calculate_uu(const cv::Mat& mat);
float calculate_vv(const cv::Mat& mat);
float calculate_uv(const cv::Mat& lhs, const cv::Mat& rhs);

void robert(const cv::Mat& img, cv::Mat& u_mat, cv::Mat& v_mat);
void xy_diff(const cv::Mat& img, cv::Mat& x_mat, cv::Mat& y_mat);

#endif