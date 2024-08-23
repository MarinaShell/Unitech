#include "cls_face_recognition.h"
#include "cls_label_gui.h"

/*****************************************************************************/
/*load model and labels*/
int cls_face_recognition::loadModel()
{
    if (_init_model)
        return 0;
    try
    {
        _init_model = true;

        // �������� ��������� ������
        _recognizer = cv::face::LBPHFaceRecognizer::create();
        _recognizer->read(_face_model);

        // ����������� ��� �� �������� �����������
        if (!_face_cascade.load(_cascade_file)) {
            throw std::runtime_error("Could not load Haar cascade");
        }
        cls_label_gui label_gui;
        if (_isNeedCrypt)
            label_gui.loadLabelGuiCrypt(_label_to_gui, _label_gui);
        else
            label_gui.loadLabelGui(_label_to_gui, _label_gui);

     }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

/*****************************************************************************/
/*recognition for all methods*/
std::string cls_face_recognition::faceRecognition(cv::Mat &image)
{
    loadModel();
    // ����������� ������� �������� (ROI)
    cv::Rect roi(_roi_x, _roi_y, _roi_width, _roi_height);
    cv::Mat roi_frame = image(roi);

    // �������������� �������� ����������� � �������� ������
    cv::Mat gray_img;
    cvtColor(roi_frame, gray_img, cv::COLOR_BGR2GRAY);


    // ����������� ��� �� ������� �����
    std::vector<cv::Rect> faces;
    _face_cascade.detectMultiScale(gray_img, 
                                    faces, 1.1, 3, 0, 
                                    cv::Size(_width_rect_face, _height_rect_face));
    std::string result_text;
    boost::uuids::uuid label;
    int label_int = -1;

    if (faces.size() > 1)
    {
        // ����������� ����� � ������ ����������� �� �����������
        result_text = "������������ ������ 1 ��������, ������ �� ��������!";
    }
    else if (!faces.empty())
    {
        // ����� ����� ������� ����
        cv::Rect largest_face = *max_element(faces.begin(), 
                                            faces.end(), 
                                            [](const cv::Rect& a, 
                                            const cv::Rect& b) 
            {
                return a.area() < b.area();
            });

        cv::Mat face = gray_img(largest_face);
        cv::resize(face, face, cv::Size(_width_rect_face, _height_rect_face)); 

        // ������������ ����� ����
        double confidence = 0.0;
        _recognizer->predict(face, label_int, confidence);

        // �������������� ����� �� int � uuid
        if (label_int != -1)
        {
            label = _label_to_gui[label_int];
        }

        // �������� ���������� �������� �����������
        if (confidence < _threshold) 
        {
            result_text = "Label: " + boost::uuids::to_string(label) + ", Confidence: " + std::to_string(confidence);
        }
        else 
        {
            result_text = "Unknown, Confidence: " + std::to_string(confidence);
        }
        largest_face.x += _roi_x;
        largest_face.y += _roi_y;
        // ��������� �������������� ������ ������ �������� ������������� ����
        rectangle(image, largest_face, cv::Scalar(255, 0, 0), 2);

        // ����������� ����� � ������ ����������� �� �����������
        putText(image, 
                result_text, 
                cv::Point(largest_face.x, largest_face.y - 10), 
                cv::FONT_HERSHEY_SIMPLEX, 
                0.9, 
                cv::Scalar(255, 0, 0), 2);
    }

    _frame = image;
    // ����������� �������� �����
    imshow("Recognized Faces", image);
    std::string uuid_str = boost::uuids::to_string(label);

    return uuid_str;
 }

/*****************************************************************************/
/*recognition foto*/
std::string cls_face_recognition::faceRecognitionImage(std::string& image_path)
{
    // �������� ����������� ��� �������������
    std::cout << "Trying to load image: " << image_path << std::endl;
    cv::Mat test_img = cv::imread(image_path, cv::IMREAD_COLOR); // ��������� ������� �����������
    if (test_img.empty())
    {
        std::cerr << "Could not open or find the image: " << image_path << std::endl;
        return "-1";
    }
    return faceRecognition(test_img);
}

/*****************************************************************************/
/*recognition camera*/
std::string cls_face_recognition::faceRecognitionCam()
{
    // ������������� ����������� � ������
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        return "-1";
    }

    cv::Mat frame;
    std::string label = "";
    while (true)
    {
        cap >> frame; // ��������� ����
        if (frame.empty()) 
        {
            std::cerr << "Error: Captured empty frame" << std::endl;
            break;
        }
 
        label = faceRecognition(frame);

        if (label != "")
            break;

        // ����� �� ����� ��� ������� ������� 'q'
        if (cv::waitKey(30) == 'q') {
            break;
        }

    }
    return label;
}

/*****************************************************************************/
/*recognition from video*/
std::string cls_face_recognition::faceRecognitionVideo()
{
     // ������������� ����������� � �����
    cv::VideoCapture cap(_video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file" << std::endl;
        return "-1";
    }

    cv::Mat frame_local;
    std::string label = "";
    while (true)
    {
        cap >> frame_local; // ��������� ����
        if (frame_local.empty())
        {
            break; // ����� ����������
        }
        label = faceRecognition(frame_local);
        if (label!="")
            break;
        // ����� �� ����� ��� ������� ������� 'q'
        if (cv::waitKey(30) == 'q') {
            break;
        }
    }

    return label;
}

/*****************************************************************************/
/*the main function to recognize face*/
std::string cls_face_recognition::faceRecognitionAll()
{
    std::promise<std::string> promise;
    std::future<std::string> future = promise.get_future();

    worker_thread = std::thread(&cls_face_recognition::faceRecognitionVideo, this);
    worker_thread.detach(); // ����������� �����, ����� �� ������� � ����

    return future.get(); // ��������� �������� �����, ���� �� ������� ���������
}

