#include "cls_scan_face.h"
#include "cls_read_image.h"
#include "cls_label_gui.h"

namespace fs = std::filesystem;

/*****************************************************************************/
/*begin scan face*/
void cls_scan_face::beginScanThread(std::promise<std::string>&& promise)
{
    try
    {
        // ������������� �����������
        cv::VideoCapture cap(_video_path);
        if (!cap.isOpened())
        {
            std::cerr << "Error: Could not open video file" << std::endl;
            promise.set_value("-1");
            return;
        }
        // ��������� ���������� ������
        int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = cap.get(cv::CAP_PROP_FPS);

        // ������� ���������� ������
        cap.set(cv::CAP_PROP_FRAME_WIDTH, frame_width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, frame_height);
        cap.set(cv::CAP_PROP_FPS, fps);

        // �������� ������� ����� ��� ����������� ���
        if (_init_cascade_haara)
        {
            if (!_face_cascade.load(_cascade_file))
            {
                std::cerr << "Error: Could not load Haar cascade" << std::endl;
                promise.set_value("-1");
                return;
            }
        }
        // �������� ������������� ���������� � ��������, ���� �� ����������
        if (!fs::exists(_output_dir))
        {
            if (fs::create_directory(_output_dir))
            {
                std::cout << "Directory created successfully: " << _output_dir << std::endl;
            }
            else
            {
                std::cerr << "Error: Could not create directory: " << _output_dir << std::endl;
                promise.set_value("-1");
                return;
            }
        }

        // �������� ������� VideoWriter ��� ������ �����
        int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G'); // ����� ��� ������

        std::string filename = _output_dir + "/" + _output_file;

        cv::VideoWriter writer(filename, codec, fps, cv::Size(frame_width, frame_height), true);

        if (!writer.isOpened())
        {
            std::cerr << "Error: Could not open video file for output" << std::endl;
            promise.set_value("-1");
            return;
        }

        // ����� ������ ������
        int64 start_time = cv::getTickCount();
        double duration = 100.0; // ����������������� � ��������

        // ���� ������� � ������ ������
        while (true)
        {
            cv::Mat frame_local;
            cap >> frame_local; // ��������� ���� � ������

            if (frame_local.empty())
            {
                std::cerr << "Error: Captured empty frame" << std::endl;
                break;
            }
            // ����������� ������� �������� (ROI)
            //cv::Rect roi(_roi_x, _roi_y, _roi_width, _roi_height);
            cv::Mat roi_frame = frame_local;// (roi);

            // �������������� �������� ����������� � �������� ������
            cv::Mat gray_img;
            cvtColor(roi_frame, gray_img, cv::COLOR_BGR2GRAY);

            // ����������� ��� � ������� ��������
            std::vector<cv::Rect> faces;
            try 
            {
                _face_cascade.detectMultiScale(gray_img,
                    faces,
                    1.1,
                    4,
                    0,
                    cv::Size(_width_rect_face, _height_rect_face)); // ����������� ���������
            }
            catch (const cv::Exception& e) 
            {
                std::cerr << "Error during detectMultiScale: " << e.what() << std::endl;
                promise.set_value("-1");
                return;
            }
   
            // �������������� ��������� ������������ ��� � ���������� ������������ ����� �����
            bool pr_is_detect_face = false;
            for (size_t i = 0; i < faces.size(); i++) 
            {
               // faces[i].x += _roi_x;
               // faces[i].y += _roi_y;
                // ��������� �������������� ������ ������������� ����
                cv::rectangle(frame_local, faces[i], cv::Scalar(255, 0, 0), 2);
                pr_is_detect_face = true;
            }
            if (!pr_is_detect_face)
                start_time = cv::getTickCount();
            int64 current_time = cv::getTickCount();
            double elapsed_time = (current_time - start_time) / cv::getTickFrequency();

            // �������������� ������� ��� ������
            std::string time_text = "Time: " + std::to_string(elapsed_time) + "s";

            // ����������� ������� �� �����
            cv::putText(frame_local, 
                        time_text, 
                        cv::Point(10, 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 
                        1, 
                        cv::Scalar(255, 255, 255), 
                        2);

            writer.write(frame_local); // �������� ���� � ���������

            imshow("Recording", frame_local); // ���������� ����
            _frame = frame_local;
            // �������� �������
            if (elapsed_time >= duration) 
            {
                break;
            }

            // ����� �� ����� ��� ������� ������� 'q'
            if (cv::waitKey(30) == 'q')
            {
                break;
            }
        }

        // ������������ ��������
        cap.release();
        writer.release();
        cv::destroyAllWindows();

        /*��������� ����� �� ������*/
        extractFramesFromVideo();
        std::string label = addPersonToModel();
        deleteDirectory();
        promise.set_value(label);

    }
    catch (const std::exception& e) 
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        promise.set_value("-1");
    }
    catch (...) 
    {
        std::cerr << "Unknown exception caught" << std::endl;
        promise.set_value("-1");
    }
}


/*****************************************************************************/
/*begin scan face*/
void beginScan()
{
    float threshold = 50.0;
    try {
        std::map<int, std::string> label_to_name;

        // ������������� ����������� � ������
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open camera" << std::endl;
            return;
        }

        // �������� ������� ��� ����������� ���
        cv::CascadeClassifier face_cascade;
        if (!face_cascade.load("./files/haarcascade_frontalface_default.xml")) {
            throw std::runtime_error("Could not load Haar cascade: haarcascade_frontalface_default.xml");
        }

        cv::Mat frame;
        while (true) {
            cap >> frame; // ��������� ����
            if (frame.empty()) {
                std::cerr << "Error: Captured empty frame" << std::endl;
                break;
            }

            // �������������� �������� ����������� � �������� ������
            cv::Mat gray_img;
            cv::cvtColor(frame, gray_img, cv::COLOR_BGR2GRAY);

            // ����������� ��� �� ������� �����
            std::vector<cv::Rect> faces;
            face_cascade.detectMultiScale(gray_img, faces, 1.1, 3, 0, cv::Size(200, 300));
            std::string result_text;

            if (faces.size() > 1) {
                // ����������� ����� � ������ ����������� �� �����������
                result_text = "������������ ������ 1 ��������, ��������!";
            }
            else if (!faces.empty()) {
                // ����� ����� ������� ����
                cv::Rect largest_face = *std::max_element(faces.begin(), faces.end(), [](const cv::Rect& a, const cv::Rect& b) {
                    return a.area() < b.area();
                    });

                cv::Mat face = gray_img(largest_face);
                cv::resize(face, face, cv::Size(200, 300)); // �������� ������ ���� �� ������������ �������

                // ��������� �������������� ������ ������ �������� ������������� ����
                cv::rectangle(frame, largest_face, cv::Scalar(255, 0, 0), 2);

                // ����������� ����� � ������ ����������� �� �����������
                //cv::putText(frame, result_text, cv::Point(largest_face.x, largest_face.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(255, 0, 0), 2);
            }

            // ����������� �������� �����
            cv::imshow("Recognized Faces", frame);

            // ����� �� ����� ��� ������� ������� 'q'
            if (cv::waitKey(30) == 'q') {
                break;
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return;
    }

    return;

    //try
    //{
    //    // ������������� �����������
    //    cv::VideoCapture cap(0);
    //    if (!cap.isOpened())
    //    {
    //        std::cerr << "Error: Could not open video file" << std::endl;
    //        return "-1";
    //    }
    //    // �������� ������� ����� ��� ����������� ���
    //    cv::CascadeClassifier face_cascade;
    //    if (!face_cascade.load("./files/haarcascade_frontalface_default.xml")) {
    //        throw std::runtime_error("Could not load Haar cascade: haarcascade_frontalface_default.xml");
    //    }

    //
    //    // ���� ������� � ������ ������
    //    cv::Mat frame_local;
    //    while (true)
    //    {
    //        cap >> frame_local; // ��������� ���� � ������

    //        if (frame_local.empty())
    //        {
    //            std::cerr << "Error: Captured empty frame" << std::endl;
    //            break;
    //        }
    //     
    //        // �������������� �������� ����������� � �������� ������
    //        cv::Mat gray_img;
    //        cvtColor(frame_local, gray_img, cv::COLOR_BGR2GRAY);

    //        // ����������� ��� � ������� ��������
    //        std::vector<cv::Rect> faces;
    //        try
    //        {
    //            face_cascade.detectMultiScale(gray_img,
    //                faces,
    //                1.1,
    //                3,
    //                0,
    //                cv::Size(100, 200)); // ����������� ���������
    //        }
    //        catch (const cv::Exception& e)
    //        {
    //            std::cerr << "Error during detectMultiScale: " << e.what() << std::endl;
    //            return "-1";
    //        }     
    //    }

    //    // ������������ ��������
    //    cap.release();
    //    cv::destroyAllWindows();

    //    return "label";
    //}
    //catch (const std::exception& e)
    //{
    //    std::cerr << "Exception caught: " << e.what() << std::endl;
    //}
    //catch (...)
    //{
    //    std::cerr << "Unknown exception caught" << std::endl;
    //}
}

/*****************************************************************************/
/* � ������*/
std::string cls_scan_face::beginScan()
{
    std::promise<std::string> promise;
    std::future<std::string> future = promise.get_future();

    worker_thread = std::thread(&cls_scan_face::beginScanThread, this, std::move(promise));
    worker_thread.detach(); // ����������� �����, ����� �� ������� � ����

    return future.get(); // ��������� �������� �����, ���� �� ������� ���������
}

/*****************************************************************************/
/*extract frames from video*/
void cls_scan_face::extractFramesFromVideo()
{
    std::string filename = _output_dir + "/" + _output_file;

    cv::VideoCapture cap(filename);
    if (!cap.isOpened()) 
    {
        std::cerr << "Error: Could not open video file" << std::endl;
        return;
    }

    int frame_number = 0;
    cv::Mat frame;
    while (true) 
    {
        cap >> frame;
        if (frame.empty()) 
            break;

        std::string frame_filename = _output_dir + 
                                    "/frame_" + 
                                    std::to_string(frame_number) + ".jpg";
        imwrite(frame_filename, frame);
        frame_number++;
    }

    cap.release();
}

/*****************************************************************************/
/*add person to model*/
bool cls_scan_face::guiExistsInMap(
    const std::map<int, boost::uuids::uuid>& label_to_uuid,
    const boost::uuids::uuid& search_uuid)
{
    return std::any_of(label_to_uuid.begin(), label_to_uuid.end(),
        [&search_uuid](const auto& pair) {
            return pair.second == search_uuid;
        });
}

/*****************************************************************************/
/*add person to model*/
std::string cls_scan_face::addPersonToModel() 
{
    std::vector<cv::Mat> images;
    std::vector<int> labels;
    std::vector<int> old_labels_int;
    std::vector<cv::Mat> old_images;
    int new_label_int = 0;
    cls_label_gui label_gui;

    // �������� ������������ ������
    cv::Ptr<cv::face::LBPHFaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();
    std::map<int, boost::uuids::uuid> label_to_gui;
    if (fs::exists(_face_model))
    {
        model->read(_face_model);
        if (_isNeedCrypt)
            label_gui.loadLabelGuiCrypt(label_to_gui, _label_gui);
        else
            label_gui.loadLabelGui(label_to_gui, _label_gui);
        old_labels_int = model->getLabels();
        old_images = model->getHistograms();
        new_label_int = old_labels_int.size();
    }
    
    // ������ ����������� ������ ��������
    cls_read_image read_image;
    read_image.setInputDirectory(_output_dir);
    read_image.readImagesWithCascadeHaar(images, labels, new_label_int);

    // ����������� ����� � ������ ������
    old_images.insert(old_images.end(), images.begin(), images.end());
    old_labels_int.insert(old_labels_int.end(), labels.begin(), labels.end());
    
    // ��������� ����������� UUID ��� �������� ��������
    boost::uuids::uuid new_label;
    
    do 
    {
        new_label = _gen();
    } 
    while (guiExistsInMap(label_to_gui, new_label));;
    
    //��������� ����� ��������
    label_to_gui[new_label_int] = new_label;
    
    // ���������� ��� �������� ������
    if (fs::exists(_face_model)) {
        model->update(old_images, old_labels_int);
    }
    else 
    {
        model->train(images, labels);
    }

    // ���������� ����������� ������
    model->save(_face_model);

    // ���������� ������ int - gui
    if (_isNeedCrypt)
        label_gui.saveLabelGuiCrypt(label_to_gui, _label_gui);
    else
        label_gui.saveLabelGui(label_to_gui, _label_gui);

    // �������������� UUID � ������
    std::string uuid_str = boost::uuids::to_string(new_label);
    return uuid_str;

    return 0;
}

/*****************************************************************************/
/*delete temp directory for recognition*/
void cls_scan_face::deleteDirectory()
{
    try 
    {
        if (fs::exists(_output_dir) && fs::is_directory(_output_dir)) 
        {
            fs::remove_all(_output_dir);
            std::cout << "Directory " << _output_dir << 
                " and all its contents were successfully deleted." << 
                std::endl;
        }
        else 
        {
            std::cerr << "Error: Directory " << _output_dir << 
                " does not exist or is not a directory." << 
                std::endl;
        }
    }
    catch (const fs::filesystem_error& e) 
    {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }
    catch (const std::exception& e) 
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return;
}
