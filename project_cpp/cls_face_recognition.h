#pragma once

#include "cls_base.h"

class cls_face_recognition:public cls_base
{
	public:
		cls_face_recognition() : stop_thread(false) {}
		~cls_face_recognition() {
			stop_thread = true;
			if (worker_thread.joinable()) {
				worker_thread.join();
			}
		}

		/*the main function to recognize face*/
		std::string faceRecognitionAll();

	private:
		int loadModel();
		/*recognition for all methods*/
		std::string faceRecognition(cv::Mat& image);
		/*recognition foto*/
		std::string faceRecognitionImage(std::string& image_path);
		/*recognition camera*/
		std::string faceRecognitionCam();
		/*recognition from video*/
		std::string faceRecognitionVideo();

		// Загрузка обученной модели
		cv::Ptr<cv::face::LBPHFaceRecognizer> _recognizer;
		cv::CascadeClassifier _face_cascade;

		float _threshold = 50.0;
		bool _init_model{ false };

		std::atomic<bool> stop_thread;
		std::thread worker_thread;

		std::map<int, boost::uuids::uuid> _label_to_gui;

};

