#include "cls_base.h"


/*****************************************************************************/
/*set adress to camera*/
void cls_base::setAdressCamera(std::string& video_path)
{
    _video_path = video_path;
}

/*****************************************************************************/
/*get video*/
cv::Mat &cls_base::getVideo()
{
    return _frame;
}

/*****************************************************************************/
/*set rect for face*/
void cls_base::setRectFace(int x, int y, int width, int height)
{
    _roi_x = x;
    _roi_y = y;
    _roi_width = width;
    _roi_height = height;
}

/*****************************************************************************/
/*need crypt*/
void cls_base::needCrypt(bool rez)
{
   _isNeedCrypt = rez;
}

