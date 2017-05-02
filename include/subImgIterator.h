#ifndef _SUB_IMG_ITERATOR
#define _SUB_IMG_ITERATOR 

#include <opencv2/opencv.hpp>
class SubImgIterator
{
public:
	typedef std::forward_iterator_tag iterator_category;
	typedef cv::Mat value_type;
	typedef cv::Point difference_type;
	typedef cv::Mat* pointer;
	typedef cv::Mat& reference;
	typedef SubImgIterator Self;

	friend bool operator==(const SubImgIterator& lhs, const SubImgIterator& rhs);

	SubImgIterator(const cv::Point& left_top,
				   cv::Mat* img,
				   size_t half_width,
				   bool bypixel=true):
	_left_top(left_top),
	_img(img),
	_half_width(half_width),
	_bypixel(bypixel)
	{
		__win_size=2*_half_width+1;
		__img_height=_img->rows;
		__img_width=_img->cols;
	}

	~SubImgIterator(){}

	SubImgIterator(const SubImgIterator& rhs):
	_left_top(rhs._left_top),
	_img(rhs._img),
	_half_width(rhs._half_width),
	_bypixel(rhs._bypixel)
	{
		__win_size=2*_half_width+1;
		__img_height=_img->rows;
		__img_width=_img->cols;
	}

	SubImgIterator(SubImgIterator&& rhs) noexcept:
	_left_top(std::move(rhs._left_top)),
	_img(rhs._img),
	_half_width(std::move(rhs._half_width)),
	_bypixel(std::move(rhs._bypixel))
	{
		__win_size=2*_half_width+1;
		__img_height=_img->rows;
		__img_width=_img->cols;
	}

	SubImgIterator& operator= (SubImgIterator rhs_copy)
	{
		using std::swap;
		swap(_left_top,rhs_copy._left_top);
		_img=rhs_copy._img;
		swap(_half_width,rhs_copy._half_width);
		swap(_bypixel,rhs_copy._bypixel);

		__win_size=2*_half_width+1;
		__img_height=_img->rows;
		__img_width=_img->cols;

		return *this;
	}

	reference operator* () 
	{
	    perpare_sub_img();
	    return __sub;
	}

	pointer operator->()
	{ 
	    return &(operator*()); 
	}

	Self& operator++()
	{
		auto x=_left_top.x;
		auto y=_left_top.y;
		if(_bypixel)
		{
			if(x+__win_size<__img_width)
			{
				x+=1;
			}
			else
			{
				x=0;
				y+=1;
			}
		}
		else
		{
			if(x+__win_size+__win_size-1<__img_width)
			{
				x+=__win_size;
			}
			else
			{
				x=0;
				y+=__win_size;
			}
		}
		_left_top.x=x;
		_left_top.y=y;
		return *this;
	}

	Self operator++(int)
	{
	    Self tmp = *this;
	    ++*this;
	    return tmp;
	}

	Self get_end()
	{
		if(_bypixel)
		{
			return Self(cv::Point(0,__img_height-__win_size+1),
						_img,_half_width,_bypixel);
		}
		else
		{
			int y_left=(int)(__img_height)%(int)(__win_size);
			int max_y=__img_height-y_left-__win_size;
			return Self(cv::Point(0,max_y+__win_size),
						_img,_half_width,_bypixel);
		}
	}

	cv::Point get_left_top()
	{
		return _left_top;
	}

	cv::Point get_pos()
	{
		return cv::Point(_left_top.x+_half_width,
						 _left_top.y+_half_width);
	}
private:

	void perpare_sub_img() 
	{

		cv::Rect rect(_left_top.x,_left_top.y,__win_size,__win_size);
		__sub=cv::Mat(*_img,rect);
	}
	cv::Point _left_top;
	size_t _half_width;
	bool _bypixel;
	cv::Mat* _img;
	cv::Mat __sub;
	size_t __win_size;
	size_t __img_height;
	size_t __img_width;
};


// SubImgIterator createSubImgIterator(cv::Mat* img,)
#endif