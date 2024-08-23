#pragma once

#include "cls_base.h"

class cls_read_image:cls_base
{
public:
    cls_read_image() : _isSaveImages(false) {}

    void readImagesWithCascadeHaar(
        std::vector<cv::Mat>& images,
        std::vector<int>& labels,
        int label);

    void setSaveFace(bool rez);

    void setInputDirectory(const std::string& directory_in);

    void setOutputDirectory(const std::string& directory_out);

private:
    bool _isSaveImages;
    std::string _directory_in;
    std::string _directory_out;
    cv::CascadeClassifier _face_cascade;
    int _init_cascade_haara = 1;

    void save_face(
        const cv::Mat& face,
        const std::string& directory,
        const std::string& base_filename,
        int index);

    bool isValidFace(const cv::Mat& face, const cv::Rect& rect);
};
