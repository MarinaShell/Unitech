#include <string>
#include "common_func.h"
//#include "cls_face_recognition.h"
//#include "cls_scan_face.h"
//void beginScan();

/*****************************************************************************/
/*main*/
using namespace cv;
using namespace cv::face;
using namespace std;
namespace fs = std::filesystem;

/*****************************************************************************/
/*load label and name from txt file*/
void loadLabel2Name(map<int,
    string>& label_to_name,
    const string& filename)
{
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Could not open file to load label to name mapping.");
    }
    int label;
    string name;
    while (file >> label >> name) {
        label_to_name[label] = name;
    }
    file.close();
}

/*****************************************************************************/
/*recognition camera*/
int face_recognition_cam()
{
    float threshold = 50.0;
    try {
        map<int, string> label_to_name;
        loadLabel2Name(label_to_name, "label_to_name.txt");

        // �������� ��������� ������
        Ptr<LBPHFaceRecognizer> recognizer = LBPHFaceRecognizer::create();
        recognizer->read("face_model.xml");

        // ������������� ����������� � ������
        VideoCapture cap(0);
        if (!cap.isOpened()) {
            cerr << "Error: Could not open camera" << endl;
            return -1;
        }

        // �������� ������� ��� ����������� ���
        CascadeClassifier face_cascade;
        if (!face_cascade.load("./files/haarcascade_frontalface_default.xml")) {
            throw runtime_error("Could not load Haar cascade: haarcascade_frontalface_default.xml");
        }

        Mat frame;
        while (true) {
            cap >> frame; // ��������� ����
            if (frame.empty()) {
                cerr << "Error: Captured empty frame" << endl;
                break;
            }

            // �������������� �������� ����������� � �������� ������
            Mat gray_img;
            cvtColor(frame, gray_img, COLOR_BGR2GRAY);

            // ����������� ��� �� ������� �����
            vector<Rect> faces;
            face_cascade.detectMultiScale(gray_img, faces, 1.1, 3, 0, Size(200, 300));
            string result_text;

            if (faces.size() > 1)
            {
                // ����������� ����� � ������ ����������� �� �����������
                result_text = "������������ ������ 1 ��������, ��������!";
            }
            else if (!faces.empty())
            {
                // ����� ����� ������� ����
                Rect largest_face = *max_element(faces.begin(), faces.end(), [](const Rect& a, const Rect& b) {
                    return a.area() < b.area();
                    });

                Mat face = gray_img(largest_face);
                resize(face, face, Size(200, 300)); // �������� ������ ���� �� ������������ �������

                // ������������ ����� ����
                int label = -1;
                double confidence = 0.0;
                recognizer->predict(face, label, confidence);

                // �������� ���������� �������� �����������
                if (confidence < threshold) {
                    result_text = "Label: " + label_to_name[label] + ", Confidence: " + to_string(confidence);
                }
                else {
                    result_text = "Unknown, Confidence: " + to_string(confidence);
                }

                // ��������� �������������� ������ ������ �������� ������������� ����
                rectangle(frame, largest_face, Scalar(255, 0, 0), 2);

                // ����������� ����� � ������ ����������� �� �����������
                putText(frame, result_text, Point(largest_face.x, largest_face.y - 10), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255, 0, 0), 2);
            }

            // ����������� �������� �����
            imshow("Recognized Faces", frame);

            // ����� �� ����� ��� ������� ������� 'q'
            if (waitKey(30) == 'q') {
                break;
            }
        }
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return -1;
    }

    return 0;
}


void main()
{

    //beginScan();
    face_recognition_cam();

    //evaluateModel("d:/marina/new_opencv/DataSetForTest");

    //class cls_scan_face scan_face;
    //// ���� � ����������
   //std::string video_path = "d:/marina/new_opencv/DataSetVideo/Sergey/bandicam 2024-05-30 11-18-52-800.mp4";
    //scan_face.setAdressCamera(video_path);
    //std::string label = scan_face.beginScan();


    //class cls_face_recognition face_recognition;
    //// ���� � ����������
    //video_path = "d:/marina/new_opencv/DataSetVideo/Sergey/bandicam 2024-05-30 11-19-00-240.mp4";

    //face_recognition.setAdressCamera(video_path);
    //label = face_recognition.faceRecognitionAll();
}