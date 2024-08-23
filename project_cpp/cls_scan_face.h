#pragma once

#include "cls_base.h"

class cls_scan_face:public cls_base
{
public:
	cls_scan_face(){}
	~cls_scan_face() {
		if (worker_thread.joinable()) {
			worker_thread.join();
		}
	}


	/*begin scan face*/
	std::string beginScan();

private:
	std::string _output_file = "tmp_video.mjpg";
	std::string _output_dir = "tmp";
	std::thread worker_thread;
	cv::CascadeClassifier _face_cascade;
	int _init_cascade_haara = 1;

	/*begin scan face*/
	void beginScanThread(std::promise<std::string>&& promise);
	bool guiExistsInMap(
		const std::map<int, boost::uuids::uuid>& label_to_uuid,
		const boost::uuids::uuid& search_uuid);

	/*extract frames from video*/
	void extractFramesFromVideo();
	/*add person to model*/
	std::string addPersonToModel();
	/*delete tmp directory*/
	void deleteDirectory();


};

