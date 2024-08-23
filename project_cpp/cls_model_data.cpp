#include "cls_model_data.h"
#include "cls_label_gui.h"
#include <boost/lexical_cast.hpp>

/*****************************************************************************/
/*get label int from array by gui*/
int cls_model_data::getLabelForGuid(
                    const std::map<int, boost::uuids::uuid>& label_to_uuid,
                    const boost::uuids::uuid& search_uuid) 
{
    for (const auto& pair : label_to_uuid) 
    {
        if (pair.second == search_uuid) 
        {
            return pair.first;
        }
    }
    return -1; // Значение по умолчанию, если UUID не найден
}

/*****************************************************************************/
/*create new model with array id*/
int cls_model_data::deleteDataById(const std::string &id_str)
{
    //переводим строку в gui
    boost::uuids::uuid id_gui;
    try 
    {
        id_gui = boost::lexical_cast<boost::uuids::uuid>(id_str);
        std::cout << "UUID: " << id_gui << std::endl;
    }
    catch (const boost::bad_lexical_cast& e) 
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    //ищем эту сроку в существующих значениях
    std::map<int, boost::uuids::uuid> label_to_gui;
    cls_label_gui label_gui;
    if (_isNeedCrypt)
        label_gui.loadLabelGuiCrypt(label_to_gui, _label_gui);
    else
        label_gui.loadLabelGui(label_to_gui, _label_gui);

    int id_int = getLabelForGuid(label_to_gui, id_gui);
    // Поиск UUID
    if (id_int>=0) 
    {
        std::cout << "GUI найден в карте" << std::endl;
    }
    else 
    {
        std::cout << "GUI не найден в карте" << std::endl;
        return -1;
    }

    // Загрузка существующей модели
    cv::Ptr<cv::face::LBPHFaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();
    model->read(_face_model);

    // Получение всех меток и изображений из модели
    std::vector<cv::Mat> all_histograms = model->getHistograms();
    std::vector<int> all_labels = model->getLabels();

    // новые метки и изображения
    std::vector<cv::Mat> filtered_images;
    std::vector<int> filtered_labels;

    for (size_t i = 0; i < all_labels.size(); ++i) 
    {
        int label = all_labels[i];
        if (label!= id_int) 
        {
            filtered_images.push_back(all_histograms[i]);
            filtered_labels.push_back(label);
        }
    }     

    // Создание новой модели с отфильтрованными данными
    cv::Ptr<cv::face::LBPHFaceRecognizer> new_model = cv::face::LBPHFaceRecognizer::create();
    new_model->train(filtered_images, filtered_labels);

    // Сохранение новой модели
    new_model->save(_face_model);

    _model->read(_face_model);

    //удаляем из карты это значение
    label_to_gui.erase(id_int);
    if (_isNeedCrypt)
        label_gui.saveLabelGuiCrypt(label_to_gui, _label_gui);
    else
        label_gui.saveLabelGui(label_to_gui, _label_gui);

    return 0;
}

/*****************************************************************************/
/*delete all model*/
int cls_model_data::deleteAllFromData()
{
    std::string file_path_face_model = _face_model;
    std::string file_path_label_gui = _label_gui;

    if (remove(file_path_face_model.c_str()) == 0 &&
        remove(file_path_label_gui.c_str()) == 0)
    {
        std::cout << "Files deleted successfully.\n";
        return 0;
    }
    else 
    {
        std::perror("Error deleting file");
        return -1;
    }
    _init_model = true;
}

/*****************************************************************************/
/*export data from model*/
int cls_model_data::exportFromData(const std::string& id_str,
    std::vector<cv::Mat> &filtered_images,
    std::vector<int> &filtered_labels,
    std::map<int, boost::uuids::uuid>& filtered_label_to_gui)
{
    //переводим строку в gui
    boost::uuids::uuid id_gui;
    try
    {
        id_gui = boost::lexical_cast<boost::uuids::uuid>(id_str);
        std::cout << "UUID: " << id_gui << std::endl;
    }
    catch (const boost::bad_lexical_cast& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    //ищем эту сроку в существующих значениях
    std::map<int, boost::uuids::uuid> label_to_gui;
    cls_label_gui label_gui;
    if (_isNeedCrypt)
        label_gui.loadLabelGuiCrypt(label_to_gui, _label_gui);
    else
        label_gui.loadLabelGui(label_to_gui, _label_gui);

    int id_int = getLabelForGuid(label_to_gui, id_gui);
    // Поиск UUID
    if (id_int >= 0)
    {
        std::cout << "GUI найден в карте" << std::endl;
    }
    else
    {
        std::cout << "GUI не найден в карте" << std::endl;
        return -1;
    }

    // Загрузка существующей модели
    if (_init_model)
    {
        _model->read(_face_model);
        _init_model = false;
    }
    // Получение всех меток и изображений из модели
    std::vector<cv::Mat> all_histograms = _model->getHistograms();
    std::vector<int> all_labels = _model->getLabels();

    for (size_t i = 0; i < all_labels.size(); ++i)
    {
        int label = all_labels[i];
        if (label == id_int)
        {
            filtered_images.push_back(all_histograms[i]);
            filtered_labels.push_back(label);
            filtered_label_to_gui[label] = id_gui;
        }
    }
    return 0;
}

/*****************************************************************************/
/* merge of 2 std::map*/
void cls_model_data::mergeMaps(
    std::map<int, boost::uuids::uuid>& dest, 
    const std::map<int, boost::uuids::uuid>& src) 
{
    for (const auto& pair : src) 
    {
        dest[pair.first] = pair.second; // Вставка элемента в карту назначения
    }
}

/*****************************************************************************/
/*import data to model*/
int cls_model_data::importToData(
    std::vector<cv::Mat>& filtered_images,
    std::vector<int>& filtered_labels,
    std::map<int, boost::uuids::uuid>& filtered_label_to_gui)
{
    cv::Ptr<cv::face::LBPHFaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();
    std::map<int, boost::uuids::uuid> label_to_gui;
    std::vector<cv::Mat> images;
    std::vector<int> labels;

    if (fs::exists(_face_model))
    {
        model->read(_face_model);
        cls_label_gui label_gui;
        if (_isNeedCrypt)
            label_gui.loadLabelGuiCrypt(label_to_gui, _label_gui);
        else
            label_gui.loadLabelGui(label_to_gui, _label_gui);
        labels = model->getLabels();
        images = model->getHistograms();
    }

    // Объединение новых и старых данных
    images.insert(images.end(), filtered_images.begin(), filtered_images.end());
    labels.insert(labels.end(), filtered_labels.begin(), filtered_labels.end());

    // Обновление или обучение модели
    if (fs::exists(_face_model)) {
        model->update(images, labels);
    }
    else
    {
        model->train(images, labels);
    }

    // Сохранение обновленной модели
    model->save(_face_model);

    // Объединение карт
    mergeMaps(label_to_gui, filtered_label_to_gui);

    // Сохранение связки int - gui
    cls_label_gui label_gui;
    if (_isNeedCrypt)
        label_gui.saveLabelGuiCrypt(label_to_gui, _label_gui);
    else
        label_gui.saveLabelGui(label_to_gui, _label_gui);
    return 0;
}

