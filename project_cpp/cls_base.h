#pragma once

#include "common_func.h"
#include <future>

class cls_base
{
public:
	/*set adress to camera*/
	void setAdressCamera(std::string& video_path);
	/*get video*/
	cv::Mat& getVideo();

	void setRectFace(int x, int y, int width, int height);

	/*need crypt*/
	void needCrypt(bool rez);


protected:
	std::string _video_path = "";
	const std::string _cascade_file = "./files/haarcascade_frontalface_default.xml";
	const int _width_rect_face = 100;
	const int _height_rect_face = 200;
	const std::string _face_model = "face_model.xml";
	const std::string _label_gui = "label_gui.bin";
	boost::uuids::random_generator _gen;
	cv::Mat _frame;
	// Размеры области интереса (ROI)
	int _roi_x = -300;
	int _roi_y = 300;
	int _roi_width = 600;
	int _roi_height = 600;
	bool _isNeedCrypt{false};
};

