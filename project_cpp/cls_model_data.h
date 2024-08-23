#pragma once

#include "cls_base.h"
class cls_model_data:cls_base

{
public:
	int exportFromData(const std::string& id,
		std::vector<cv::Mat>& images,
		std::vector<int>& labels,
		std::map<int, boost::uuids::uuid>& label_to_gui);
	int importToData(
		std::vector<cv::Mat>& images,
		std::vector<int>& labels,
		std::map<int, boost::uuids::uuid>& label_to_gui);
	/*delete person from model by id*/
	int deleteDataById(const std::string &id);
	/*delete all model*/
	int deleteAllFromData();

private:
	enum class EAct { add, del };
	cv::Ptr<cv::face::LBPHFaceRecognizer> _model;
	bool _init_model = true;

	int getLabelForGuid(
		const std::map<int, boost::uuids::uuid>& label_to_uuid,
		const boost::uuids::uuid& search_uuid);

	/* Функция для объединения двух std::map*/
	void mergeMaps(std::map<int, boost::uuids::uuid>& dest,
		const std::map<int, boost::uuids::uuid>& src);
};

