#include "cls_read_image.h"

/********************************************************************************/
/*save face if true*/
void cls_read_image::save_face(
    const cv::Mat& face,
    const std::string& directory,
    const std::string& base_filename,
    int index)
{
    std::string filename = directory + "/" + 
                            base_filename + "_face_" + 
                            std::to_string(index) + ".jpg";
    imwrite(filename, face);
    std::cout << "Saved face to file: " << filename << std::endl;
}


/********************************************************************************/
/*if face id Valid*/
bool cls_read_image::isValidFace(const cv::Mat& face, const cv::Rect& rect)
{
    double aspect_ratio = (double)rect.width / rect.height;
    double mean_intensity = mean(face(rect))[0];
    return mean_intensity > 50 && mean_intensity < 200;
}

/********************************************************************************/
/*set input directory fot input images*/
void cls_read_image::setInputDirectory(const std::string& directory_in)
{
    _directory_in = directory_in;
}

/********************************************************************************/
/*set output directory fot input images*/
void cls_read_image::setOutputDirectory(const std::string& directory_out)
{
    _directory_out = directory_out;
}

/********************************************************************************/
/*save images*/
void cls_read_image::setSaveFace(bool rez)
{
    _isSaveImages = rez;
}

/********************************************************************************/
/*read images with Cascade Haar*/
void cls_read_image::readImagesWithCascadeHaar(
    std::vector<cv::Mat>& images,
    std::vector<int>& labels, 
    int label)
{
    if (_init_cascade_haara)
    {
        if (!_face_cascade.load(_cascade_file))
        {
            throw std::runtime_error("Could not load Haar cascade");
        }
        _init_cascade_haara = 0;
    }
 
    int k = 0; // Счетчик для имен файлов
    if (_isSaveImages)
        fs::create_directory(_directory_out);

    for (const auto& person_dir : fs::directory_iterator(_directory_in))
    {
        if (fs::is_directory(person_dir))
        {
            std::string person_name = person_dir.path().filename().string(); // Имя директории

            for (const auto& entry : fs::directory_iterator(person_dir.path()))
            {
                std::string path = entry.path().string();
                cv::Mat img = cv::imread(path, cv::IMREAD_COLOR); // Загружаем изображение в цвете
                if (!img.empty())
                {
                    cv::Mat gray_img;
                    cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);

                    // Обнаружение лиц
                    std::vector<cv::Rect> faces;
                    _face_cascade.detectMultiScale(gray_img,
                        faces,
                        1.1,
                        4,
                        0,
                        cv::Size(_width_rect_face, _height_rect_face)); // Настроенные параметры

                    if (!faces.empty()) 
                    {
                        // Фильтрация и нахождение самого большого лица
                        cv::Rect largest_face;
                        bool found_valid_face = false;
                        for (const cv::Rect& face : faces)
                        {
                            if (isValidFace(gray_img, face) &&
                                (!found_valid_face || face.area() > largest_face.area()))
                            {
                                largest_face = face;
                                found_valid_face = true;
                            }
                        }

                        if (found_valid_face)
                        {
                            // Вырезать самое большое лицо
                            cv::Mat face = gray_img(largest_face);
                            images.push_back(face);
                            labels.push_back(label);

                            // Сохранить лицо
                            if (_isSaveImages)
                            {
                                save_face(face, _directory_out, "temp", k);
                                k++;
                                std::cout << "Loaded face from image: " << path << std::endl;
                            }
                        }
                        else
                        {
                            std::cerr << "No valid faces found in image: " << path << std::endl;
                        }
                    }
                    else
                    {
                        std::cerr << "No faces found in image: " << path << std::endl;
                    }
                }
                else
                {
                    std::cerr << "Could not read image: " << path << std::endl;
                }
            }
        }
        else
        {
            std::cerr << "Skipping non-directory: " << person_dir.path() << std::endl;
        }
        label++;
    }

    if (images.empty())
    {
        std::cerr << "No faces were loaded from the dataset "<< std::endl; 
    }
}
