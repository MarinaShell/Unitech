#include "cls_read_image.h"
#include "cls_label_2_name.h"


void evaluateModel(const string& test_directory)
{
    try 
    {
        map<int, string> label_to_name;
        cls_label_2_name::loadLabel2Name(label_to_name, "label_to_name.txt");

        // «агрузка обученной модели
        Ptr<LBPHFaceRecognizer> recognizer = LBPHFaceRecognizer::create();
        recognizer->read("face_model.xml");

        vector<Mat> test_images;
        vector<int> true_labels;
        map<int, string> temp_map; // временное отображение дл€ хранени€ имен в read_images

        static cls_read_image read_images;
        read_images.setInputDirectory(test_directory);
        read_images.readImagesWithCascadeHaar(test_images, true_labels, temp_map);

        int correct_predictions = 0;
        unordered_map<int, unordered_map<int, int>> confusion_matrix;
        for (const auto& entry : label_to_name) 
        {
            confusion_matrix[entry.first] = unordered_map<int, int>();
            for (const auto& inner_entry : label_to_name) 
            {
                confusion_matrix[entry.first][inner_entry.first] = 0;
            }
        }

        for (size_t i = 0; i < test_images.size(); i++) 
        {
            Mat face = test_images[i];

            // ѕредсказание метки лица
            int predicted_label = -1;
            double confidence = 0.0;
            recognizer->predict(face, predicted_label, confidence);

            confusion_matrix[true_labels[i]][predicted_label]++;

            if (predicted_label == true_labels[i]) 
            {
                correct_predictions++;
            }
        }

        double accuracy = (double)correct_predictions / test_images.size();
        cout << "Accuracy: " << accuracy * 100.0 << "%" << endl;

        // ¬ычисление precision, recall и F1-меры дл€ каждого класса
        for (const auto& entry : label_to_name)
        {
            int label = entry.first;
            int true_positive = confusion_matrix[label][label];
            int false_positive = 0;
            int false_negative = 0;
            int true_negative = 0;

            for (const auto& inner_entry : label_to_name)
            {
                if (inner_entry.first != label) 
                {
                    false_positive += confusion_matrix[inner_entry.first][label];
                    false_negative += confusion_matrix[label][inner_entry.first];
                }
            }

            int total_predictions = 0;
            for (const auto& inner_entry : label_to_name)
            {
                total_predictions += confusion_matrix[inner_entry.first][label];
            }

            true_negative = test_images.size() - (true_positive + false_positive + false_negative);

            double precision = (true_positive + false_positive == 0) ? 0 : (double)true_positive / (true_positive + false_positive);
            double recall = (true_positive + false_negative == 0) ? 0 : (double)true_positive / (true_positive + false_negative);
            double f1_score = (precision + recall == 0) ? 0 : 2 * (precision * recall) / (precision + recall);
            
            cout << "Class: " << entry.second << endl;
            cout << "Precision: " << precision * 100.0 << "%" << endl;
            cout << "Recall: " << recall * 100.0 << "%" << endl;
            cout << "F1 Score: " << f1_score * 100.0 << "%" << endl;
            cout << "--------------------------" << endl;
        }
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
    }
}