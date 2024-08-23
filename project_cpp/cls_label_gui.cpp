#include "cls_label_gui.h"
#include <boost/lexical_cast.hpp>

/*****************************************************************************/
/*save label and name to bin file*/
void cls_label_gui::saveLabelGui(const std::map<int, boost::uuids::uuid>& label_to_gui,
                                 const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file to save label to name mapping.");
    }
    for (const auto& entry : label_to_gui)
    {
        file << entry.first << " " << entry.second << std::endl;
    }
    file.close();
}

/*****************************************************************************/
/*load label and name from txt file*/
void cls_label_gui::loadLabelGui(std::map<int, boost::uuids::uuid>& label_to_gui,
                                 const std::string& filename)
{
    
    if (!fs::exists(filename))
        return;
    std::ifstream file(filename);
    
    if (!file.is_open()) 
    {
        throw std::runtime_error("Could not open file to load label to name mapping.");
    }
    int label;
    boost::uuids::uuid gui;
    while (file >> label >> gui) 
    {
        label_to_gui[label] = gui;
    }
    file.close();
}

/*****************************************************************************/
/*save label and name to crypt file*/
void cls_label_gui::saveLabelGuiCrypt(const std::map<int, boost::uuids::uuid>& label_to_gui,
                                      const std::string& filename)
{

    // Сериализация карты
    std::string serialized_data = serializeMap(label_to_gui);

    // Шифрование сериализованных данных
    std::string encrypted_data = cls_crypt::encrypt(serialized_data, _key, _iv);

    std::ofstream file(filename, std::ios::binary);
    file.write(encrypted_data.c_str(), encrypted_data.size());
    file.close();
}

/*****************************************************************************/
/*load label and name from crypt file*/
void cls_label_gui::loadLabelGuiCrypt(std::map<int, boost::uuids::uuid>& label_to_gui,
                                             const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    std::stringstream buffer;
    buffer << file.rdbuf();

    // Чтение зашифрованных данных из файла
    std::string read_encrypted_data = buffer.str();

    // Дешифрование данных
    std::string decrypted_data = cls_crypt::decrypt(read_encrypted_data, _key, _iv);

    // Десериализация карты
    label_to_gui = deserializeMap(decrypted_data);
}

/*****************************************************************************/
/*load label and name from crypt file*/
void cls_label_gui::setKeyForCrypt(
    const CryptoPP::byte key[CryptoPP::AES::DEFAULT_KEYLENGTH],
    const CryptoPP::byte iv[CryptoPP::AES::BLOCKSIZE])
{

    std::copy(key, key + CryptoPP::AES::DEFAULT_KEYLENGTH, _key);

    std::copy(iv, iv + CryptoPP::AES::BLOCKSIZE, _iv);
}

/*****************************************************************************/
/*function for serialize map int - gui*/
std::string cls_label_gui::serializeMap(const std::map<int, boost::uuids::uuid>& map)
{
    std::ostringstream oss;
    for (const auto& pair : map)
    {
        oss << pair.first << " " << boost::uuids::to_string(pair.second) << "\n";
    }
    return oss.str();
}

/*****************************************************************************/
/*function for deserialize map int - gui*/
std::map<int, boost::uuids::uuid> cls_label_gui::deserializeMap(const std::string& data)
{
    std::map<int, boost::uuids::uuid> map;
    std::istringstream iss(data);
    int key;
    std::string uuid_str;
    while (iss >> key >> uuid_str) {
        map[key] = boost::lexical_cast<boost::uuids::uuid>(uuid_str);
    }
    return map;
}